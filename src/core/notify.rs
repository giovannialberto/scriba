//! Desktop notifications for Scriba.
//!
//! Fires native OS notifications by shelling out to the platform's tooling so
//! we don't need to pull in a notification crate (or its GUI dependencies).
//! - macOS: `osascript` (always available).
//! - Linux: `notify-send` from libnotify (commonly installed).
//!
//! On any failure or unsupported platform this is a best-effort no-op.

use anyhow::Result;
use std::process::Command;
use std::time::Duration;

/// Native notification panel helper (macOS).
///
/// Unbundled CLI processes cannot post Notification Center banners with
/// action buttons and `display dialog` is an unstyled centered box, so Scriba
/// draws its own notification-style panel (top-right, vibrancy, buttons) via
/// a small AppKit helper. The Swift source is embedded and compiled once with
/// `swiftc` (ships with the Xcode Command Line Tools, which Homebrew
/// requires); everything falls back to AppleScript when it is unavailable.
#[cfg(target_os = "macos")]
mod panel {
    use anyhow::Result;
    use std::path::PathBuf;

    const SOURCE: &str = include_str!("notify_panel.swift");

    fn cache_dir() -> Option<PathBuf> {
        let home = std::env::var_os("HOME")?;
        Some(PathBuf::from(home).join("Library/Caches/scriba"))
    }

    /// Helper path, keyed by a hash of the embedded source so upgrades
    /// recompile automatically.
    fn helper_path() -> Option<PathBuf> {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        SOURCE.hash(&mut hasher);
        Some(cache_dir()?.join(format!("notify-panel-{:016x}", hasher.finish())))
    }

    /// Path to the compiled helper, if it has been built already.
    pub fn existing() -> Option<PathBuf> {
        helper_path().filter(|p| p.exists())
    }

    /// Compile the helper if needed. Idempotent; call at startup so the first
    /// notification doesn't wait ~10s on swiftc.
    pub fn ensure() -> Result<PathBuf> {
        let path = helper_path().ok_or_else(|| anyhow::anyhow!("no home directory"))?;
        if path.exists() {
            return Ok(path);
        }
        let dir = path.parent().expect("helper path has a parent");
        std::fs::create_dir_all(dir)?;
        // Helpers built by other Scriba versions are left alone: another
        // instance (an older install, a dev build in a second terminal) may
        // still be running and would lose its panel, falling back to
        // AppleScript. They are ~130 KB each.
        let source_path = path.with_extension("swift");
        std::fs::write(&source_path, SOURCE)?;
        let output = std::process::Command::new("swiftc")
            .args(["-O", "-o"])
            .arg(&path)
            .arg(&source_path)
            .output()?;
        let _ = std::fs::remove_file(&source_path);
        if !output.status.success() {
            return Err(anyhow::anyhow!(
                "swiftc failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        Ok(path)
    }
}

/// Whether the native notification panel is already built (always true on
/// platforms that don't need one).
pub fn notification_helper_ready() -> bool {
    #[cfg(target_os = "macos")]
    {
        panel::existing().is_some()
    }
    #[cfg(not(target_os = "macos"))]
    {
        true
    }
}

/// Build the native notification panel helper ahead of time so the first
/// meeting notification doesn't wait on a compile. Blocking (runs swiftc on
/// first call, ~10s); no-op off macOS. On failure callers keep working — the
/// AppleScript fallbacks are used instead.
pub fn prepare_notification_helper() -> Result<()> {
    #[cfg(target_os = "macos")]
    panel::ensure()?;
    Ok(())
}

/// Fire a desktop notification.
///
/// `title` is the bold headline and `body` is the subtext. Sound is played on
/// platforms that support it. Failures are swallowed and logged to stderr so a
/// notification hiccup never breaks the watcher loop.
pub fn notify(title: &str, body: &str) {
    if let Err(e) = try_notify(title, body) {
        eprintln!("⚠️  Desktop notification failed: {e}");
    }
}

fn try_notify(title: &str, body: &str) -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        if let Some(helper) = panel::existing() {
            let mut child = Command::new(helper)
                .args([
                    "notify",
                    "--title",
                    title,
                    "--subtitle",
                    body,
                    "--timeout",
                    "5",
                    "--sound",
                    "Glass",
                ])
                .stdout(std::process::Stdio::null())
                .spawn()?;
            // The panel outlives this call by design; reap it off-thread so it
            // doesn't linger as a zombie.
            std::thread::spawn(move || {
                let _ = child.wait();
            });
            return Ok(());
        }
        // Escape double quotes for the AppleScript string literal.
        let title_esc = title.replace('\\', "\\\\").replace('"', "\\\"");
        let body_esc = body.replace('\\', "\\\\").replace('"', "\\\"");
        let script = format!(
            "display notification \"{body_esc}\" with title \"{title_esc}\" sound name \"Glass\""
        );
        let status = Command::new("osascript").args(["-e", &script]).status()?;
        if !status.success() {
            anyhow::bail!("osascript exited with status {status}");
        }
    }

    #[cfg(target_os = "linux")]
    {
        let status = Command::new("notify-send")
            .args(["--icon", "audio-input-microphone", title, body])
            .status()?;
        if !status.success() {
            anyhow::bail!("notify-send exited with status {status}");
        }
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        eprintln!("🔔 {title}: {body}");
    }

    Ok(())
}

/// Show a desktop confirmation with a yes/no choice and return the answer.
/// `message` is the second line under the title (keep it short; the native
/// panel renders it as a subtitle and resolves bundle IDs to app names).
///
/// Blocks (async) until the user chooses, the dialog times out, or the future
/// is dropped (the dialog process is killed on drop, so callers can cancel it
/// via `select!`). On timeout or any failure (e.g. no dialog tooling), falls
/// back to `default_answer` — a plain notification is fired instead on
/// failure so the event is not silently swallowed.
///
/// - macOS: native notification-style panel (top-right, pill buttons). There
///   is deliberately no `display dialog` fallback (a centered modal): without
///   the panel, a plain notification is fired and `default_answer` is used.
/// - Linux: `notify-send -A` action buttons (libnotify 0.7.9+).
pub async fn confirm(
    title: &str,
    message: &str,
    yes_label: &str,
    no_label: &str,
    timeout_secs: u32,
    default_answer: bool,
) -> bool {
    let attempt = try_confirm(
        title,
        message,
        yes_label,
        no_label,
        timeout_secs,
        default_answer,
    );
    // Belt over the tooling's own timeout in case it isn't honored. Generous:
    // hovering the native panel legitimately pauses its countdown while the
    // user decides.
    let hard_timeout = Duration::from_secs(timeout_secs as u64 + 600);
    match tokio::time::timeout(hard_timeout, attempt).await {
        Ok(Ok(answer)) => answer,
        // No stderr here: the caller may be hosted by the TUI. The plain
        // notification is the visible fallback.
        Ok(Err(_)) => {
            notify(title, message);
            default_answer
        }
        Err(_) => default_answer,
    }
}

#[cfg(target_os = "macos")]
async fn try_confirm(
    title: &str,
    message: &str,
    yes_label: &str,
    no_label: &str,
    timeout_secs: u32,
    default_answer: bool,
) -> Result<bool> {
    // Native notification-style panel, top-right with action buttons.
    if let Some(helper) = panel::existing() {
        let output = tokio::process::Command::new(helper)
            .args([
                "confirm",
                "--title",
                title,
                "--subtitle",
                message,
                "--yes",
                yes_label,
                "--no",
                no_label,
                "--timeout",
                &timeout_secs.to_string(),
                "--sound",
                "Glass",
            ])
            .kill_on_drop(true)
            .output()
            .await?;
        if !output.status.success() {
            anyhow::bail!("notify panel exited with status {}", output.status);
        }
        return Ok(match String::from_utf8_lossy(&output.stdout).trim() {
            "yes" => true,
            "no" => false,
            _ => default_answer,
        });
    }

    // No centered `display dialog` fallback: it is an unstyled modal in the
    // middle of the screen. Without the panel the caller fires a plain
    // notification and uses the default answer.
    anyhow::bail!("native notification panel is not built")
}

#[cfg(target_os = "linux")]
async fn try_confirm(
    title: &str,
    message: &str,
    yes_label: &str,
    no_label: &str,
    timeout_secs: u32,
    default_answer: bool,
) -> Result<bool> {
    // `-A` prints the chosen action's key to stdout and exits; on timeout or
    // dismissal it exits with empty output.
    let output = tokio::process::Command::new("notify-send")
        .args([
            "--app-name",
            "Scriba",
            "--icon",
            "audio-input-microphone",
            "-A",
            &format!("yes={yes_label}"),
            "-A",
            &format!("no={no_label}"),
            "-t",
            &(timeout_secs * 1000).to_string(),
            title,
            message,
        ])
        .kill_on_drop(true)
        .output()
        .await?;
    if !output.status.success() {
        anyhow::bail!("notify-send exited with status {}", output.status);
    }
    Ok(match String::from_utf8_lossy(&output.stdout).trim() {
        "yes" => true,
        "no" => false,
        _ => default_answer,
    })
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
async fn try_confirm(
    _title: &str,
    _message: &str,
    _yes_label: &str,
    _no_label: &str,
    _timeout_secs: u32,
    default_answer: bool,
) -> Result<bool> {
    Ok(default_answer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notify_does_not_panic_on_long_strings() {
        // Should never panic, regardless of content.
        notify(&"x".repeat(500), &"y".repeat(500));
    }
}
