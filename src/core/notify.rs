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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notify_does_not_panic_on_long_strings() {
        // Should never panic, regardless of content.
        notify(&"x".repeat(500), &"y".repeat(500));
    }
}
