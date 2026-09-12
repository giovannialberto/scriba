//! Voice enrollment: record the owner reading a few sentences and turn the
//! clip into voice samples that diarization recognizes. Shared by onboarding
//! and the Settings screen.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use crossterm::event::KeyCode;
use futures_util::FutureExt;
use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use super::chat::ACCENT;
use crate::core::{RecordOptions, ScribaConfig, record_audio};
use crate::database::Database;

/// How long the owner reads for.
pub(super) const ENROLL_SECS: f32 = 12.0;
/// Enter stops the recording early once this much has been captured.
const MIN_EARLY_STOP_SECS: f32 = 4.0;
/// If the recorder has not returned this long after being told to stop, the
/// microphone is stuck (permissions, missing device) and we give up.
const STOP_GRACE_SECS: f32 = 8.0;

pub(super) enum VoicePhase {
    /// Showing the script, waiting for Enter.
    Prompt,
    Recording {
        started: Instant,
        level: f32,
        stop_tx: mpsc::Sender<()>,
        level_rx: mpsc::Receiver<f32>,
        task: JoinHandle<Result<()>>,
        stop_sent: bool,
        stop_at: Option<Instant>,
    },
    Processing {
        task: JoinHandle<Result<(usize, usize, f32)>>,
    },
    Done {
        stored: usize,
        total_samples: usize,
        total_secs: f32,
    },
    Failed(String),
    Skipped,
}

/// Outcome of a key press.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum VoiceAction {
    Continue,
    /// The flow is over (done, failed, or skipped): the caller moves on.
    Finished,
}

pub(super) struct VoiceEnrollment {
    pub(super) phase: VoicePhase,
    name: String,
    dir: PathBuf,
}

impl VoiceEnrollment {
    pub(super) fn new(name: &str) -> Self {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        Self {
            phase: VoicePhase::Prompt,
            name: name.trim().to_string(),
            dir: std::env::temp_dir().join(format!("scriba-voice-{stamp}")),
        }
    }

    /// The sentences to read: short, natural, and covering varied sounds.
    pub(super) fn script(&self) -> Vec<String> {
        let hello = if self.name.is_empty() {
            "\u{201C}Hey Scriba, it's me.".to_string()
        } else {
            format!("\u{201C}Hey Scriba, this is {}.", self.name)
        };
        vec![
            hello,
            "I'm going to use you to remember my meetings,".to_string(),
            "my ideas, and the people I work with.".to_string(),
            "Let's see how well you learn my voice.\u{201D}".to_string(),
        ]
    }

    pub(super) fn handle_key(&mut self, key: KeyCode, config: &ScribaConfig) -> VoiceAction {
        match (&mut self.phase, key) {
            (VoicePhase::Prompt, KeyCode::Enter) => {
                self.start_recording(config);
                VoiceAction::Continue
            }
            (VoicePhase::Prompt, KeyCode::Char('s') | KeyCode::Char('S')) => {
                self.phase = VoicePhase::Skipped;
                VoiceAction::Finished
            }
            (
                VoicePhase::Recording {
                    started,
                    stop_tx,
                    stop_sent,
                    stop_at,
                    ..
                },
                KeyCode::Enter,
            ) => {
                if !*stop_sent && started.elapsed().as_secs_f32() >= MIN_EARLY_STOP_SECS {
                    let _ = stop_tx.try_send(());
                    *stop_sent = true;
                    *stop_at = Some(Instant::now());
                }
                VoiceAction::Continue
            }
            (
                VoicePhase::Done { .. } | VoicePhase::Failed(_) | VoicePhase::Skipped,
                KeyCode::Enter,
            ) => VoiceAction::Finished,
            (VoicePhase::Failed(_), KeyCode::Char('r') | KeyCode::Char('R')) => {
                self.phase = VoicePhase::Prompt;
                VoiceAction::Continue
            }
            _ => VoiceAction::Continue,
        }
    }

    fn start_recording(&mut self, config: &ScribaConfig) {
        let (stop_tx, stop_rx) = mpsc::channel::<()>(1);
        let (level_tx, level_rx) = mpsc::channel::<f32>(64);
        let dir = self.dir.clone();
        let input_device = config.audio_settings.input_device.clone();
        let task = tokio::spawn(async move {
            let options = RecordOptions {
                compression_settings: None,
                stop_rx: Some(stop_rx),
                level_tx: Some(level_tx),
                verbose: false,
                silence_timeout: None,
                input_device,
                loopback_device: None,
                register_in_db: false,
            };
            record_audio(dir, options).await.map(|_| ())
        });
        self.phase = VoicePhase::Recording {
            started: Instant::now(),
            level: 0.0,
            stop_tx,
            level_rx,
            task,
            stop_sent: false,
            stop_at: None,
        };
    }

    /// Advance timers and background work; call once per frame.
    pub(super) fn tick(&mut self) {
        match &mut self.phase {
            VoicePhase::Recording {
                started,
                level,
                stop_tx,
                level_rx,
                task,
                stop_sent,
                stop_at,
            } => {
                while let Ok(l) = level_rx.try_recv() {
                    *level = l;
                }
                if !*stop_sent && started.elapsed().as_secs_f32() >= ENROLL_SECS {
                    let _ = stop_tx.try_send(());
                    *stop_sent = true;
                    *stop_at = Some(Instant::now());
                }
                if let Some(at) = stop_at
                    && at.elapsed().as_secs_f32() > STOP_GRACE_SECS
                    && !task.is_finished()
                {
                    task.abort();
                    let _ = std::fs::remove_dir_all(&self.dir);
                    self.phase = VoicePhase::Failed(
                        "The microphone did not respond. Check that Scriba may use the microphone \
                         (System Settings > Privacy & Security > Microphone) and try again."
                            .to_string(),
                    );
                    return;
                }
                if task.is_finished() {
                    let dir = self.dir.clone();
                    let handle = std::mem::replace(task, tokio::spawn(async { Ok(()) }));
                    self.phase = VoicePhase::Processing {
                        task: tokio::spawn(async move {
                            match handle.await {
                                Ok(Ok(())) => {}
                                Ok(Err(e)) => return Err(e),
                                Err(e) => {
                                    return Err(anyhow::anyhow!("recording task failed: {e}"));
                                }
                            }
                            let clip = dir.join("recording.wav");
                            let result = async {
                                let samples =
                                    crate::core::voice::owner_samples_from_file(&clip).await?;
                                let mut db = Database::new()?;
                                crate::core::voice::store_owner_samples(
                                    &mut db,
                                    &samples,
                                    "enrollment",
                                )?;
                                let (count, secs) = db.speaker_sample_stats("owner", crate::core::diarization::EMBEDDING_MODEL_ID)?;
                                Ok::<_, anyhow::Error>((samples.len(), count, secs))
                            }
                            .await;
                            let _ = std::fs::remove_dir_all(&dir);
                            result
                        }),
                    };
                }
            }
            VoicePhase::Processing { task } => {
                if task.is_finished() {
                    let handle = std::mem::replace(task, tokio::spawn(async { Ok((0, 0, 0.0)) }));
                    // The handle is finished, so polling it once yields the result.
                    self.phase = match handle.now_or_never() {
                        Some(Ok(Ok((stored, total_samples, total_secs)))) => VoicePhase::Done {
                            stored,
                            total_samples,
                            total_secs,
                        },
                        Some(Ok(Err(e))) => VoicePhase::Failed(format!("{e:#}")),
                        Some(Err(e)) => VoicePhase::Failed(format!("{e}")),
                        None => VoicePhase::Failed("voice processing did not finish".to_string()),
                    };
                }
            }
            _ => {}
        }
    }

    /// Footer hint for the current phase.
    pub(super) fn footer_hint(&self) -> &'static str {
        match self.phase {
            VoicePhase::Prompt => "[Enter] Record  [S] Skip",
            VoicePhase::Recording { .. } => "[Enter] Stop early",
            VoicePhase::Processing { .. } => "Learning your voice...",
            VoicePhase::Failed(_) => "[R] Retry  [Enter] Continue",
            VoicePhase::Done { .. } | VoicePhase::Skipped => "[Enter] Continue",
        }
    }

    /// Body lines for the current phase. `width` is the available columns.
    pub(super) fn render_lines(&self, width: usize) -> Vec<Line<'static>> {
        let white = Style::default().fg(Color::White);
        let dim = Style::default().fg(Color::Indexed(245));
        let accent = Style::default().fg(ACCENT).add_modifier(Modifier::BOLD);
        let mut lines: Vec<Line<'static>> = Vec::new();
        match &self.phase {
            VoicePhase::Prompt => {
                lines.push(Line::from(Span::styled(
                    "Let Scriba learn your voice.",
                    white,
                )));
                lines.push(Line::from(""));
                lines.push(Line::from(Span::styled(
                    format!(
                        "Press Enter, then read this aloud (about {} seconds):",
                        ENROLL_SECS as u32
                    ),
                    white,
                )));
                lines.push(Line::from(""));
                for s in self.script() {
                    lines.push(Line::from(Span::styled(format!("   {s}"), accent)));
                }
                lines.push(Line::from(""));
                lines.push(Line::from(Span::styled(
                    "Only a voiceprint is kept, never the audio. You can redo this in Settings.",
                    dim,
                )));
            }
            VoicePhase::Recording {
                started,
                level,
                stop_sent,
                ..
            } => {
                let elapsed = started.elapsed().as_secs_f32().min(ENROLL_SECS);
                let remaining = (ENROLL_SECS - elapsed).ceil() as u32;
                lines.push(Line::from(vec![
                    Span::styled(
                        "\u{25CF} Recording ",
                        Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        if *stop_sent {
                            "finishing...".to_string()
                        } else {
                            format!("{remaining}s left")
                        },
                        white,
                    ),
                ]));
                lines.push(Line::from(""));
                for s in self.script() {
                    lines.push(Line::from(Span::styled(format!("   {s}"), accent)));
                }
                lines.push(Line::from(""));
                let bar_width = width.saturating_sub(12).clamp(10, 40);
                let filled = ((elapsed / ENROLL_SECS) * bar_width as f32) as usize;
                let progress = format!(
                    "{}{}",
                    "\u{2593}".repeat(filled),
                    "\u{2591}".repeat(bar_width - filled)
                );
                lines.push(Line::from(vec![
                    Span::styled("   time  ", dim),
                    Span::styled(progress, Style::default().fg(ACCENT)),
                ]));
                let lvl = ((level * 4.0).min(1.0) * bar_width as f32) as usize;
                let meter = format!(
                    "{}{}",
                    "\u{25AE}".repeat(lvl),
                    "\u{25AF}".repeat(bar_width - lvl)
                );
                lines.push(Line::from(vec![
                    Span::styled("   level ", dim),
                    Span::styled(meter, Style::default().fg(Color::Green)),
                ]));
            }
            VoicePhase::Processing { .. } => {
                lines.push(Line::from(Span::styled("Learning your voice...", white)));
                lines.push(Line::from(Span::styled(
                    "First time only: downloading the voice model (about 40 MB).",
                    dim,
                )));
            }
            VoicePhase::Done {
                stored,
                total_samples,
                total_secs,
            } => {
                lines.push(Line::from(Span::styled(
                    format!("\u{2713} Got it. Learned {stored} sample(s) of your voice."),
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                )));
                lines.push(Line::from(Span::styled(
                    format!("Scriba now knows {total_samples} sample(s), {total_secs:.0}s in total, and keeps learning from your meetings."),
                    dim,
                )));
            }
            VoicePhase::Failed(msg) => {
                lines.push(Line::from(Span::styled(
                    "\u{2717} Could not learn your voice.",
                    Style::default().fg(Color::Red),
                )));
                lines.push(Line::from(Span::styled(msg.clone(), dim)));
            }
            VoicePhase::Skipped => {
                lines.push(Line::from(Span::styled(
                    "Skipped. You can teach Scriba your voice later in Settings.",
                    dim,
                )));
            }
        }
        lines
    }
}

impl std::fmt::Debug for VoiceEnrollment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let phase = match self.phase {
            VoicePhase::Prompt => "prompt",
            VoicePhase::Recording { .. } => "recording",
            VoicePhase::Processing { .. } => "processing",
            VoicePhase::Done { .. } => "done",
            VoicePhase::Failed(_) => "failed",
            VoicePhase::Skipped => "skipped",
        };
        write!(f, "VoiceEnrollment({phase})")
    }
}

impl Drop for VoiceEnrollment {
    fn drop(&mut self) {
        match &self.phase {
            VoicePhase::Recording { task, stop_tx, .. } => {
                let _ = stop_tx.try_send(());
                task.abort();
            }
            VoicePhase::Processing { task } => task.abort(),
            _ => {}
        }
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn script_uses_the_name_and_prompt_renders() {
        let v = VoiceEnrollment::new("Giovanni");
        assert!(v.script()[0].contains("Giovanni"));
        let lines = v.render_lines(80);
        assert!(lines.len() > 5);
        assert_eq!(v.footer_hint(), "[Enter] Record  [S] Skip");
        let anon = VoiceEnrollment::new("   ");
        assert!(anon.script()[0].contains("it's me"));
    }

    #[test]
    fn skip_finishes_immediately() {
        let mut v = VoiceEnrollment::new("G");
        let config = ScribaConfig::default();
        assert_eq!(
            v.handle_key(KeyCode::Char('s'), &config),
            VoiceAction::Finished
        );
        assert!(matches!(v.phase, VoicePhase::Skipped));
        assert_eq!(v.footer_hint(), "[Enter] Continue");
    }
}
