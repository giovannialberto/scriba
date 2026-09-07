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

/// Show a desktop confirmation with a yes/no choice and return the answer.
///
/// Blocks (async) until the user chooses, the dialog times out, or the future
/// is dropped (the dialog process is killed on drop, so callers can cancel it
/// via `select!`). On timeout or any failure (e.g. no dialog tooling), falls
/// back to `default_answer` — a plain notification is fired instead on
/// failure so the event is not silently swallowed.
///
/// - macOS: `osascript` `display dialog` with two buttons and `giving up
///   after` the timeout.
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
    // Belt over the tooling's own timeout in case it isn't honored.
    let hard_timeout = Duration::from_secs(timeout_secs as u64 + 10);
    match tokio::time::timeout(hard_timeout, attempt).await {
        Ok(Ok(answer)) => answer,
        Ok(Err(e)) => {
            eprintln!("⚠️  Confirmation dialog failed ({e}); assuming default");
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
    let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
    let script = format!(
        "display dialog \"{}\" with title \"{}\" buttons {{\"{}\", \"{}\"}} default button \"{}\" giving up after {}",
        esc(message),
        esc(title),
        esc(no_label),
        esc(yes_label),
        esc(yes_label),
        timeout_secs
    );
    let output = tokio::process::Command::new("osascript")
        .args(["-e", &script])
        .kill_on_drop(true)
        .output()
        .await?;
    if !output.status.success() {
        anyhow::bail!("osascript exited with status {}", output.status);
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    if stdout.contains("gave up:true") {
        return Ok(default_answer);
    }
    Ok(stdout.contains(&format!("button returned:{yes_label}")))
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
