//! Where account secrets live.
//!
//! macOS: the login Keychain, through the `keyring` crate, under the
//! service name "scriba". Everything else: a file with mode 0600 under
//! `~/scriba_recordings/.secrets/`, which is no worse than where API keys
//! already sit in `config.json`. If the Keychain is unavailable (headless
//! session, locked keychain) the file backend is used as a fallback so
//! sign-in never fails because of secret storage.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// The Supabase refresh token of the signed-in Scriba Pro account.
pub const SESSION_TOKEN: &str = "cloud-session";

#[cfg(target_os = "macos")]
const KEYCHAIN_SERVICE: &str = "scriba";

fn secrets_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().context("Failed to get home directory")?;
    Ok(home.join("scriba_recordings").join(".secrets"))
}

/// Store `value` under `name`, replacing any previous value.
pub fn store(name: &str, value: &str) -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        if let Ok(entry) = keyring::Entry::new(KEYCHAIN_SERVICE, name)
            && entry.set_password(value).is_ok()
        {
            // A stale file copy must not shadow the keychain later.
            let _ = file_delete(&secrets_dir()?, name);
            return Ok(());
        }
    }
    file_store(&secrets_dir()?, name, value)
}

/// Read the value stored under `name`, if any.
pub fn load(name: &str) -> Result<Option<String>> {
    #[cfg(target_os = "macos")]
    {
        if let Ok(entry) = keyring::Entry::new(KEYCHAIN_SERVICE, name) {
            match entry.get_password() {
                Ok(value) => return Ok(Some(value)),
                Err(keyring::Error::NoEntry) => {}
                Err(_) => {} // locked or unavailable: try the file
            }
        }
    }
    file_load(&secrets_dir()?, name)
}

/// Remove `name` from every backend. Missing entries are not an error.
pub fn delete(name: &str) -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        if let Ok(entry) = keyring::Entry::new(KEYCHAIN_SERVICE, name) {
            let _ = entry.delete_credential();
        }
    }
    file_delete(&secrets_dir()?, name)
}

/// Whether anything is stored under `name`.
pub fn exists(name: &str) -> bool {
    matches!(load(name), Ok(Some(_)))
}

// ─── File backend ────────────────────────────────────────────────────────────

fn file_path(dir: &Path, name: &str) -> PathBuf {
    dir.join(name)
}

pub(crate) fn file_store(dir: &Path, name: &str, value: &str) -> Result<()> {
    fs::create_dir_all(dir).context("Failed to create secrets directory")?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(dir, fs::Permissions::from_mode(0o700));
    }
    let path = file_path(dir, name);
    fs::write(&path, value).context("Failed to write secret")?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600))
            .context("Failed to restrict secret permissions")?;
    }
    Ok(())
}

pub(crate) fn file_load(dir: &Path, name: &str) -> Result<Option<String>> {
    let path = file_path(dir, name);
    if !path.exists() {
        return Ok(None);
    }
    let value = fs::read_to_string(&path).context("Failed to read secret")?;
    let value = value.trim_end_matches(['\n', '\r']).to_string();
    Ok(if value.is_empty() { None } else { Some(value) })
}

pub(crate) fn file_delete(dir: &Path, name: &str) -> Result<()> {
    let path = file_path(dir, name);
    match fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e).context("Failed to delete secret"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(test: &str) -> PathBuf {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("scriba-secrets-{test}-{stamp}"))
    }

    #[test]
    fn file_backend_round_trips_and_deletes() {
        let dir = temp_dir("roundtrip");
        assert_eq!(file_load(&dir, "x").unwrap(), None);
        file_store(&dir, "x", "token-1\n").unwrap();
        assert_eq!(file_load(&dir, "x").unwrap().as_deref(), Some("token-1"));
        file_store(&dir, "x", "token-2").unwrap();
        assert_eq!(file_load(&dir, "x").unwrap().as_deref(), Some("token-2"));
        file_delete(&dir, "x").unwrap();
        file_delete(&dir, "x").unwrap();
        assert_eq!(file_load(&dir, "x").unwrap(), None);
        let _ = fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn file_backend_is_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let dir = temp_dir("mode");
        file_store(&dir, "y", "v").unwrap();
        let mode = fs::metadata(dir.join("y")).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
        let _ = fs::remove_dir_all(&dir);
    }
}
