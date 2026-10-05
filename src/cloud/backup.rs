//! Scriba Pro backups: a snapshot of the knowledge layer (and optionally the
//! audio) in the account's cloud storage, and the way back.
//!
//! Content-addressed: every file is uploaded once under its SHA-256 and a
//! manifest maps relative paths to hashes. The client talks to the proxy's
//! `/backup/v1` endpoints only to list, sign and commit; bytes go straight
//! to storage through short-lived signed URLs. This is backup and restore,
//! not sync: one active device at a time, guarded by the manifest
//! generation.

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{CloudError, SESSION_HINT, access_token, proxy_url};
use crate::core::ScribaConfig;
use crate::database::Database;

/// Audio extensions, excluded unless the user opted in.
const AUDIO_EXTENSIONS: &[&str] = &[
    "wav", "mp3", "m4a", "aac", "flac", "ogg", "opus", "aiff", "aif", "webm", "caf",
];
/// Top-level files that belong to the knowledge layer.
const TOP_LEVEL_FILES: &[&str] = &["world.md", "scriba_mcp.json"];
/// Files per signing request.
const BATCH: usize = 200;
/// Parallel transfers to object storage.
const PARALLEL: usize = 6;
/// Files up to this size are sent as one in-memory body; larger ones stream.
const BUFFERED_UPLOAD_MAX: u64 = 64 * 1024 * 1024;

/// One entry of a manifest, local or remote.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ManifestFile {
    pub path: String,
    pub sha256: String,
    pub size: u64,
    #[serde(default)]
    pub modified: String,
}

/// The remote state of a backup.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct Manifest {
    #[serde(default)]
    pub generation: u64,
    #[serde(default)]
    pub updated_at: String,
    #[serde(default)]
    pub bytes: u64,
    #[serde(default)]
    pub files: Vec<ManifestFile>,
    #[serde(default)]
    pub client_version: String,
}

impl Manifest {
    /// Recording directories referenced by the manifest.
    pub fn recording_count(&self) -> usize {
        let mut dirs: Vec<&str> = self
            .files
            .iter()
            .filter_map(|f| f.path.split_once('/').map(|(d, _)| d))
            .collect();
        dirs.sort_unstable();
        dirs.dedup();
        dirs.len()
    }

    /// Whether any audio file is part of the backup.
    pub fn has_audio(&self) -> bool {
        self.files.iter().any(|f| is_audio(Path::new(&f.path)))
    }
}

/// A file on this machine that belongs in the backup.
#[derive(Debug, Clone)]
pub struct LocalFile {
    /// Path inside the backup (forward slashes, relative).
    pub path: String,
    /// Where to read the bytes from (a staging copy for the DB and config).
    pub source: PathBuf,
    pub sha256: String,
    pub size: u64,
    pub modified: String,
}

impl LocalFile {
    fn manifest_entry(&self) -> ManifestFile {
        ManifestFile {
            path: self.path.clone(),
            sha256: self.sha256.clone(),
            size: self.size,
            modified: self.modified.clone(),
        }
    }
}

/// Progress reported while a backup or restore runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Progress {
    Scanning,
    /// `done` of `total` files transferred so far, in bytes too.
    Transferring {
        done: usize,
        total: usize,
        bytes_done: u64,
        bytes_total: u64,
    },
    Committing,
}

/// Result of a backup run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackupReport {
    pub manifest: Manifest,
    pub uploaded: usize,
    pub uploaded_bytes: u64,
    pub unchanged: usize,
}

fn is_audio(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| AUDIO_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// Recording directories are named `YYYY-MM-DD_HH-MM-SS_<name>`.
fn is_recording_dir(name: &str) -> bool {
    let b = name.as_bytes();
    b.len() > 20
        && b[..4].iter().all(u8::is_ascii_digit)
        && b[4] == b'-'
        && b[5..7].iter().all(u8::is_ascii_digit)
        && b[7] == b'-'
        && b[8..10].iter().all(u8::is_ascii_digit)
        && b[10] == b'_'
}

fn skip_file(name: &str) -> bool {
    name.starts_with('.')
        || name.ends_with(".tmp")
        || name.ends_with(".part")
        || name.ends_with(".bak")
}

fn sha256_file(path: &Path) -> Result<(String, u64)> {
    let mut file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    let mut size = 0u64;
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        size += n as u64;
    }
    Ok((format!("{:x}", hasher.finalize()), size))
}

fn modified_rfc3339(path: &Path) -> String {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .map(|t| chrono::DateTime::<chrono::Utc>::from(t).to_rfc3339())
        .unwrap_or_default()
}

/// Strip every credential from the config before it leaves the machine.
/// The account section is kept out too: it is per device.
pub fn sanitize_config(json: &str) -> Result<String> {
    let mut v: serde_json::Value =
        serde_json::from_str(json).context("config.json is not valid JSON")?;
    if let Some(o) = v.as_object_mut() {
        o.remove("last_api_key");
        o.insert("stt_api_keys".into(), serde_json::json!({}));
        o.remove("cloud");
        if let Some(api) = o.get_mut("transcription").and_then(|t| t.get_mut("Api")) {
            api["api_key"] = serde_json::Value::String(String::new());
        }
        if let Some(e) = o.get_mut("enrichment").and_then(|e| e.as_object_mut()) {
            e.insert("cloud_api_keys".into(), serde_json::json!({}));
            if let Some(cloud) = e.get_mut("mode").and_then(|m| m.get_mut("Cloud")) {
                cloud["api_key"] = serde_json::Value::String(String::new());
            }
        }
    }
    Ok(serde_json::to_string_pretty(&v)?)
}

/// Everything that should be in the backup, hashed. `staging` receives the
/// sanitized config and a consistent snapshot of the database.
pub fn scan_local(base: &Path, include_audio: bool, staging: &Path) -> Result<Vec<LocalFile>> {
    std::fs::create_dir_all(staging)?;
    let mut files = Vec::new();
    let mut push = |path: String, source: PathBuf| -> Result<()> {
        let (sha256, size) = sha256_file(&source)?;
        let modified = modified_rfc3339(&source);
        files.push(LocalFile {
            path,
            source,
            sha256,
            size,
            modified,
        });
        Ok(())
    };

    for name in TOP_LEVEL_FILES {
        let p = base.join(name);
        if p.is_file() {
            push((*name).to_string(), p)?;
        }
    }

    let config_path = base.join("config.json");
    if config_path.is_file() {
        let staged = staging.join("config.json");
        std::fs::write(
            &staged,
            sanitize_config(&std::fs::read_to_string(&config_path)?)?,
        )?;
        push("config.json".to_string(), staged)?;
    }

    let db_path = base.join("scriba.db");
    if db_path.is_file() {
        let staged = staging.join("scriba.db");
        let _ = std::fs::remove_file(&staged);
        Database::snapshot_to(&db_path, &staged).context("snapshot the database")?;
        push("scriba.db".to_string(), staged)?;
    }

    let mut dirs: Vec<PathBuf> = std::fs::read_dir(base)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.is_dir()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .map(is_recording_dir)
                    .unwrap_or(false)
        })
        .collect();
    dirs.sort();
    for dir in dirs {
        let dir_name = dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_string();
        let mut entries: Vec<PathBuf> = std::fs::read_dir(&dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_file())
            .collect();
        entries.sort();
        for file in entries {
            let Some(name) = file.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if skip_file(name) || (!include_audio && is_audio(&file)) {
                continue;
            }
            push(format!("{dir_name}/{name}"), file)?;
        }
    }
    Ok(files)
}

/// Files that are not already in the remote manifest with the same content.
pub fn plan_uploads<'a>(local: &'a [LocalFile], remote: Option<&Manifest>) -> Vec<&'a LocalFile> {
    let remote: HashMap<&str, &str> = remote
        .map(|m| {
            m.files
                .iter()
                .map(|f| (f.path.as_str(), f.sha256.as_str()))
                .collect()
        })
        .unwrap_or_default();
    local
        .iter()
        .filter(|f| remote.get(f.path.as_str()) != Some(&f.sha256.as_str()))
        .collect()
}

// ─── Wire client ─────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct UploadsResponse {
    #[serde(default)]
    uploads: Vec<SignedUpload>,
}

#[derive(Debug, Clone, Deserialize)]
struct SignedUpload {
    sha256: String,
    url: String,
    #[serde(default)]
    headers: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct DownloadsResponse {
    #[serde(default)]
    downloads: Vec<SignedDownload>,
}

#[derive(Debug, Clone, Deserialize)]
struct SignedDownload {
    sha256: String,
    url: String,
}

/// Client for the proxy's backup endpoints and for the signed storage URLs.
/// Uses the newer HTTP client (`reqwest13`): the crate's default one stalls
/// on streamed request bodies.
#[derive(Debug, Clone)]
pub struct BackupApi {
    base: String,
    http: reqwest13::Client,
}

impl BackupApi {
    /// `None` when the proxy is not configured in this build.
    pub fn from_config(config: &ScribaConfig) -> Option<Self> {
        let proxy = proxy_url(config)?;
        let http = reqwest13::Client::builder()
            .timeout(Duration::from_secs(600))
            .connect_timeout(Duration::from_secs(20))
            .redirect(reqwest13::redirect::Policy::none())
            .build()
            .ok()?;
        Some(Self {
            base: format!("{proxy}/backup/v1"),
            http,
        })
    }

    fn token() -> Result<String, CloudError> {
        access_token().ok_or_else(|| CloudError::Other(SESSION_HINT.to_string()))
    }

    async fn send(&self, req: reqwest13::RequestBuilder) -> Result<(u16, String), CloudError> {
        let resp = req
            .bearer_auth(Self::token()?)
            .send()
            .await
            .map_err(|e| CloudError::Network(e.to_string()))?;
        let status = resp.status().as_u16();
        let body = resp.text().await.unwrap_or_default();
        Ok((status, body))
    }

    fn error_from(status: u16, body: &str) -> CloudError {
        let v: serde_json::Value = serde_json::from_str(body).unwrap_or_default();
        let message = v["error"]["message"].as_str().unwrap_or("").to_string();
        let kind = v["error"]["type"].as_str().unwrap_or("").to_string();
        match (status, kind.as_str()) {
            (401, _) | (403, _) => CloudError::SessionExpired,
            (413, _) | (_, "quota_exceeded") => CloudError::Rejected(if message.is_empty() {
                "backup storage quota exceeded".into()
            } else {
                message
            }),
            (409, _) => CloudError::Rejected(if message.is_empty() {
                "another device changed the backup; try again".into()
            } else {
                message
            }),
            (503, _) => CloudError::Other("backups are not available right now".into()),
            _ => CloudError::Other(if message.is_empty() {
                format!("backup request failed ({status})")
            } else {
                message
            }),
        }
    }

    /// The remote manifest, or `None` when the account has no backup yet.
    pub async fn manifest(&self) -> Result<Option<Manifest>, CloudError> {
        let (status, body) = self
            .send(self.http.get(format!("{}/manifest", self.base)))
            .await?;
        match status {
            200 => serde_json::from_str(&body)
                .map(Some)
                .map_err(|e| CloudError::Other(format!("bad manifest: {e}"))),
            404 => Ok(None),
            _ => Err(Self::error_from(status, &body)),
        }
    }

    async fn uploads(&self, files: &[ManifestFile]) -> Result<Vec<SignedUpload>, CloudError> {
        let body = serde_json::json!({ "files": files });
        let (status, text) = self
            .send(self.http.post(format!("{}/uploads", self.base)).json(&body))
            .await?;
        if status != 200 {
            return Err(Self::error_from(status, &text));
        }
        let parsed: UploadsResponse = serde_json::from_str(&text)
            .map_err(|e| CloudError::Other(format!("bad uploads: {e}")))?;
        Ok(parsed.uploads)
    }

    async fn commit(
        &self,
        expected_generation: Option<u64>,
        files: &[ManifestFile],
    ) -> Result<Manifest, CloudError> {
        let body = serde_json::json!({
            "expected_generation": expected_generation,
            "client_version": super::client_version(),
            "files": files,
        });
        let (status, text) = self
            .send(self.http.post(format!("{}/commit", self.base)).json(&body))
            .await?;
        if status != 200 {
            return Err(Self::error_from(status, &text));
        }
        serde_json::from_str(&text)
            .map_err(|e| CloudError::Other(format!("bad commit response: {e}")))
    }

    async fn downloads(&self, sha256s: &[String]) -> Result<Vec<SignedDownload>, CloudError> {
        let body = serde_json::json!({ "sha256s": sha256s });
        let (status, text) = self
            .send(
                self.http
                    .post(format!("{}/downloads", self.base))
                    .json(&body),
            )
            .await?;
        if status != 200 {
            return Err(Self::error_from(status, &text));
        }
        let parsed: DownloadsResponse = serde_json::from_str(&text)
            .map_err(|e| CloudError::Other(format!("bad downloads: {e}")))?;
        Ok(parsed.downloads)
    }

    /// Delete the account's backup entirely.
    pub async fn delete_all(&self) -> Result<(), CloudError> {
        let (status, text) = self.send(self.http.delete(&self.base)).await?;
        if status == 204 || status == 200 || status == 404 {
            Ok(())
        } else {
            Err(Self::error_from(status, &text))
        }
    }

    async fn put_file(&self, upload: &SignedUpload, source: &Path) -> Result<()> {
        let len = tokio::fs::metadata(source).await?.len();
        let body = if len <= BUFFERED_UPLOAD_MAX {
            reqwest13::Body::from(tokio::fs::read(source).await?)
        } else {
            let file = tokio::fs::File::open(source).await?;
            reqwest13::Body::wrap_stream(tokio_util::io::ReaderStream::new(file))
        };
        let mut req = self
            .http
            .put(&upload.url)
            .header(reqwest13::header::CONTENT_LENGTH, len)
            .body(body);
        for (k, v) in &upload.headers {
            req = req.header(k.as_str(), v.as_str());
        }
        let resp = req.send().await.context("upload")?;
        if !resp.status().is_success() {
            bail!("storage refused the upload ({})", resp.status());
        }
        Ok(())
    }

    async fn get_to_file(&self, url: &str, dest: &Path, expected_sha256: &str) -> Result<()> {
        use tokio::io::AsyncWriteExt;
        let mut resp = self.http.get(url).send().await.context("download")?;
        if !resp.status().is_success() {
            bail!("storage refused the download ({})", resp.status());
        }
        if let Some(parent) = dest.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let tmp = dest.with_extension("part");
        let mut out = tokio::fs::File::create(&tmp).await?;
        let mut hasher = Sha256::new();
        while let Some(chunk) = resp.chunk().await? {
            hasher.update(&chunk);
            out.write_all(&chunk).await?;
        }
        out.flush().await?;
        drop(out);
        let got = format!("{:x}", hasher.finalize());
        if got != expected_sha256 {
            let _ = tokio::fs::remove_file(&tmp).await;
            bail!("downloaded file did not match its checksum");
        }
        tokio::fs::rename(&tmp, dest).await?;
        Ok(())
    }
}

// ─── Backup ──────────────────────────────────────────────────────────────────

/// Run a backup of `base`. `progress` is called from the current task.
pub async fn run_backup(
    api: &BackupApi,
    base: &Path,
    include_audio: bool,
    mut progress: impl FnMut(Progress),
) -> Result<BackupReport> {
    progress(Progress::Scanning);
    let staging = std::env::temp_dir().join(format!("scriba-backup-{}", std::process::id()));
    let local = {
        let base = base.to_path_buf();
        let staging = staging.clone();
        tokio::task::spawn_blocking(move || scan_local(&base, include_audio, &staging))
            .await
            .context("scan task")??
    };
    let result = backup_scanned(api, &local, &mut progress).await;
    let _ = std::fs::remove_dir_all(&staging);
    result
}

async fn backup_scanned(
    api: &BackupApi,
    local: &[LocalFile],
    progress: &mut impl FnMut(Progress),
) -> Result<BackupReport> {
    let remote = api.manifest().await?;
    let to_upload = plan_uploads(local, remote.as_ref());
    let bytes_total: u64 = to_upload.iter().map(|f| f.size).sum();
    let total = to_upload.len();
    let mut done = 0usize;
    let mut bytes_done = 0u64;
    progress(Progress::Transferring {
        done,
        total,
        bytes_done,
        bytes_total,
    });

    for chunk in to_upload.chunks(BATCH) {
        let entries: Vec<ManifestFile> = chunk.iter().map(|f| f.manifest_entry()).collect();
        let signed = api.uploads(&entries).await?;
        let by_hash: HashMap<&str, &SignedUpload> =
            signed.iter().map(|u| (u.sha256.as_str(), u)).collect();
        // Not signed means already stored: nothing to send for those. Each
        // job owns its data so the futures carry no borrows.
        let jobs: Vec<(String, PathBuf, u64, Option<SignedUpload>)> = chunk
            .iter()
            .map(|file| {
                (
                    file.path.clone(),
                    file.source.clone(),
                    file.size,
                    by_hash.get(file.sha256.as_str()).map(|u| (*u).clone()),
                )
            })
            .collect();
        let mut transfers = futures_util::stream::iter(jobs.into_iter().map(|(path, source, size, upload)| {
            let api = api.clone();
            async move {
                if let Some(upload) = upload {
                    api.put_file(&upload, &source)
                        .await
                        .with_context(|| format!("upload {path}"))?;
                }
                Ok::<u64, anyhow::Error>(size)
            }
        }))
        .buffer_unordered(PARALLEL);
        while let Some(result) = transfers.next().await {
            let size = result?;
            done += 1;
            bytes_done += size;
            progress(Progress::Transferring {
                done,
                total,
                bytes_done,
                bytes_total,
            });
        }
    }

    progress(Progress::Committing);
    let entries: Vec<ManifestFile> = local.iter().map(|f| f.manifest_entry()).collect();
    let manifest = api
        .commit(remote.as_ref().map(|m| m.generation), &entries)
        .await?;
    Ok(BackupReport {
        manifest,
        uploaded: total,
        uploaded_bytes: bytes_total,
        unchanged: local.len() - total,
    })
}

// ─── Restore ─────────────────────────────────────────────────────────────────

/// What a restore brought back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestoreReport {
    pub files: usize,
    pub bytes: u64,
    pub recordings: usize,
}

/// Restore `manifest` into `base`. Files land next to the live data: the
/// database goes to `scriba.db.restored` and the config to
/// `config.json.restored`; call [`apply_restored`] afterwards (it needs the
/// open database and the current config).
pub async fn run_restore(
    api: &BackupApi,
    base: &Path,
    manifest: &Manifest,
    mut progress: impl FnMut(Progress),
) -> Result<RestoreReport> {
    let total = manifest.files.len();
    let bytes_total: u64 = manifest.files.iter().map(|f| f.size).sum();
    let mut done = 0usize;
    let mut bytes_done = 0u64;
    progress(Progress::Transferring {
        done,
        total,
        bytes_done,
        bytes_total,
    });
    for chunk in manifest.files.chunks(BATCH) {
        let hashes: Vec<String> = chunk.iter().map(|f| f.sha256.clone()).collect();
        let signed = api.downloads(&hashes).await?;
        let by_hash: HashMap<&str, &SignedDownload> =
            signed.iter().map(|d| (d.sha256.as_str(), d)).collect();
        let mut jobs: Vec<(String, PathBuf, String, String, u64)> = Vec::with_capacity(chunk.len());
        for file in chunk {
            let dest = restore_destination(base, &file.path)?;
            let url = by_hash
                .get(file.sha256.as_str())
                .map(|d| d.url.clone())
                .ok_or_else(|| anyhow::anyhow!("no download for {}", file.path))?;
            jobs.push((file.path.clone(), dest, url, file.sha256.clone(), file.size));
        }
        let mut transfers = futures_util::stream::iter(jobs.into_iter().map(|(path, dest, url, sha, size)| {
            let api = api.clone();
            async move {
                api.get_to_file(&url, &dest, &sha)
                    .await
                    .with_context(|| format!("restore {path}"))?;
                Ok::<u64, anyhow::Error>(size)
            }
        }))
        .buffer_unordered(PARALLEL);
        while let Some(result) = transfers.next().await {
            let size = result?;
            done += 1;
            bytes_done += size;
            progress(Progress::Transferring {
                done,
                total,
                bytes_done,
                bytes_total,
            });
        }
    }
    Ok(RestoreReport {
        files: total,
        bytes: bytes_total,
        recordings: manifest.recording_count(),
    })
}

/// Where a manifest path is written during restore. Rejects anything that
/// would escape `base` even if a hostile manifest tried.
fn restore_destination(base: &Path, path: &str) -> Result<PathBuf> {
    if path.is_empty()
        || path.starts_with('/')
        || path
            .split('/')
            .any(|seg| seg.is_empty() || seg == "." || seg == "..")
        || path.contains('\\')
    {
        bail!("refusing to restore suspicious path {path:?}");
    }
    let rel = match path {
        "scriba.db" => "scriba.db.restored".to_string(),
        "config.json" => "config.json.restored".to_string(),
        other => other.to_string(),
    };
    Ok(base.join(rel))
}

/// Load the restored database and config into place. The database is
/// copied into the open connection through SQLite's backup API, so no file
/// swap happens under a live connection; the config keeps this device's
/// account section.
pub fn apply_restored(base: &Path, db: &mut Database, config: &mut ScribaConfig) -> Result<()> {
    let db_file = base.join("scriba.db.restored");
    if db_file.is_file() {
        db.restore_from_file(&db_file)
            .context("load the restored database")?;
        let _ = std::fs::remove_file(&db_file);
    }
    let cfg_file = base.join("config.json.restored");
    if cfg_file.is_file() {
        let text = std::fs::read_to_string(&cfg_file)?;
        let mut restored: ScribaConfig =
            serde_json::from_str(&text).context("the restored config does not parse")?;
        restored.enrichment.migrate_legacy();
        restored.cloud = config.cloud.clone();
        *config = restored;
        config.save()?;
        let _ = std::fs::remove_file(&cfg_file);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_base(tag: &str) -> PathBuf {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let p = std::env::temp_dir().join(format!("scriba-backup-test-{tag}-{stamp}"));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn config_sanitizer_strips_every_credential_and_the_account() {
        let raw = r#"{
          "transcription": {"Api": {"api_key": "sk-stt", "base_url": null, "model": null}},
          "last_api_key": "sk-old",
          "stt_api_keys": {"openai": "sk-1"},
          "enrichment": {"mode": {"Cloud": {"provider": "Anthropic", "api_key": "sk-ant", "model": null, "base_url": null}}, "cloud_api_keys": {"anthropic": "sk-ant"}},
          "cloud": {"email": "me@x.io"},
          "check_for_updates": true
        }"#;
        let out = sanitize_config(raw).unwrap();
        assert!(!out.contains("sk-"), "{out}");
        assert!(!out.contains("me@x.io"));
        assert!(out.contains("check_for_updates"));
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["transcription"]["Api"]["api_key"], "");
        assert_eq!(v["enrichment"]["mode"]["Cloud"]["api_key"], "");
    }

    #[test]
    fn scan_picks_the_knowledge_layer_and_skips_audio_by_default() {
        let base = temp_base("scan");
        std::fs::write(base.join("world.md"), "# world").unwrap();
        std::fs::write(
            base.join("config.json"),
            r#"{"stt_api_keys":{"openai":"sk"}}"#,
        )
        .unwrap();
        std::fs::write(base.join(".DS_Store"), "x").unwrap();
        std::fs::create_dir_all(base.join("models")).unwrap();
        std::fs::write(base.join("models/big.onnx"), "model").unwrap();
        let rec = base.join("2026-10-05_17-02-29_Meeting");
        std::fs::create_dir_all(&rec).unwrap();
        std::fs::write(rec.join("recording.mp3"), "audio").unwrap();
        std::fs::write(rec.join("transcript.txt"), "hello").unwrap();
        std::fs::write(rec.join("transcript.txt.tmp"), "partial").unwrap();
        let staging = base.join(".staging");

        let files = scan_local(&base, false, &staging).unwrap();
        let paths: Vec<&str> = files.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(
            paths,
            vec![
                "world.md",
                "config.json",
                "2026-10-05_17-02-29_Meeting/transcript.txt"
            ]
        );
        let cfg = files.iter().find(|f| f.path == "config.json").unwrap();
        assert!(
            !std::fs::read_to_string(&cfg.source)
                .unwrap()
                .contains("\"sk\"")
        );
        assert_eq!(files[0].sha256.len(), 64);

        let with_audio = scan_local(&base, true, &staging).unwrap();
        assert!(with_audio.iter().any(|f| f.path.ends_with("recording.mp3")));
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn upload_plan_skips_unchanged_files() {
        let local = vec![
            LocalFile {
                path: "world.md".into(),
                source: PathBuf::new(),
                sha256: "a".into(),
                size: 1,
                modified: String::new(),
            },
            LocalFile {
                path: "scriba.db".into(),
                source: PathBuf::new(),
                sha256: "b".into(),
                size: 2,
                modified: String::new(),
            },
            LocalFile {
                path: "r/transcript.txt".into(),
                source: PathBuf::new(),
                sha256: "c".into(),
                size: 3,
                modified: String::new(),
            },
        ];
        let remote = Manifest {
            files: vec![
                ManifestFile {
                    path: "world.md".into(),
                    sha256: "a".into(),
                    size: 1,
                    modified: String::new(),
                },
                ManifestFile {
                    path: "scriba.db".into(),
                    sha256: "old".into(),
                    size: 2,
                    modified: String::new(),
                },
            ],
            ..Default::default()
        };
        let plan = plan_uploads(&local, Some(&remote));
        let paths: Vec<&str> = plan.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(paths, vec!["scriba.db", "r/transcript.txt"]);
        assert_eq!(plan_uploads(&local, None).len(), 3);
        assert_eq!(remote.recording_count(), 0);
    }

    #[test]
    fn restore_paths_stay_inside_the_base() {
        let base = Path::new("/tmp/base");
        assert_eq!(
            restore_destination(base, "scriba.db").unwrap(),
            base.join("scriba.db.restored")
        );
        assert_eq!(
            restore_destination(base, "2026-01-01_00-00-00_x/transcript.txt").unwrap(),
            base.join("2026-01-01_00-00-00_x/transcript.txt")
        );
        for bad in ["../etc/passwd", "/abs", "a//b", "a/./b", "a\\b", ""] {
            assert!(restore_destination(base, bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn manifests_describe_themselves() {
        let m = Manifest {
            files: vec![
                ManifestFile {
                    path: "world.md".into(),
                    sha256: "a".into(),
                    size: 1,
                    modified: String::new(),
                },
                ManifestFile {
                    path: "d1/transcript.txt".into(),
                    sha256: "b".into(),
                    size: 1,
                    modified: String::new(),
                },
                ManifestFile {
                    path: "d1/recording.mp3".into(),
                    sha256: "c".into(),
                    size: 1,
                    modified: String::new(),
                },
                ManifestFile {
                    path: "d2/transcript.txt".into(),
                    sha256: "d".into(),
                    size: 1,
                    modified: String::new(),
                },
            ],
            ..Default::default()
        };
        assert_eq!(m.recording_count(), 2);
        assert!(m.has_audio());
        assert!(is_recording_dir("2026-10-05_17-02-29_Meeting"));
        assert!(!is_recording_dir("models"));
        assert!(!is_recording_dir("2026-10-05"));
    }
}
