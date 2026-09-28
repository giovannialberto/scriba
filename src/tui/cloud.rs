//! Scriba Pro account flow, shown inline under the Account row in Settings.
//!
//! Signed out it offers two things: ask to join the closed beta, or sign in
//! with an email code. Signed in it shows the entitlements and lets the user
//! refresh them or sign out. Network work runs in a task; `tick` collects the
//! result and hands the caller an [`AccountEvent`] to apply to the config.

use crossterm::event::KeyCode;
use futures_util::FutureExt;
use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
};
use tokio::task::JoinHandle;

use super::chat::ACCENT;
use crate::cloud::{self, AccountEvent, AccountStatus, CloudError, SupabaseClient};
use crate::core::ScribaConfig;

/// Longest note accepted with a beta request (matches the server).
const NOTE_MAX: usize = 500;
/// Codes are six digits today; leave room for longer ones.
const CODE_MAX: usize = 10;
const CODE_MIN: usize = 6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Purpose {
    Request,
    SignIn,
}

/// What a background task is doing, so a failure knows where to go back to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Op {
    Request { email: String },
    SendCode { email: String },
    Verify { email: String },
    Refresh,
    SignOut,
}

pub(super) enum CloudPhase {
    /// This build has no project configured.
    Unavailable,
    /// Signed out: choose to request access or sign in.
    Menu,
    Email {
        purpose: Purpose,
        buffer: String,
    },
    Note {
        email: String,
        buffer: String,
    },
    Code {
        email: String,
        buffer: String,
    },
    Busy {
        op: Op,
        task: JoinHandle<Result<AccountEvent, CloudError>>,
    },
    Requested {
        email: String,
    },
    /// Signed in.
    Account,
    Failed {
        message: String,
        back: Box<CloudPhase>,
    },
}

/// Outcome of a key press.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CloudAction {
    Continue,
    /// The flow is over: the caller closes it.
    Finished,
}

pub(super) struct CloudFlow {
    pub(super) phase: CloudPhase,
    client: Option<SupabaseClient>,
    /// Signed-in email, or the last one typed, to prefill the field.
    email_hint: String,
    entitlements: Vec<String>,
    requested_at: Option<String>,
}

impl CloudFlow {
    pub(super) fn new(config: &ScribaConfig) -> Self {
        let client = cloud::client_for(config);
        let (phase, email_hint, entitlements, requested_at) = match cloud::status(config) {
            AccountStatus::Unavailable => (CloudPhase::Unavailable, String::new(), vec![], None),
            AccountStatus::SignedOut { requested_at } => {
                (CloudPhase::Menu, String::new(), vec![], requested_at)
            }
            AccountStatus::SignedIn {
                email,
                entitlements,
            } => (CloudPhase::Account, email, entitlements, None),
        };
        Self {
            phase,
            client,
            email_hint,
            entitlements,
            requested_at,
        }
    }

    /// Keep the flow's view of the account in step with the config after an
    /// event was applied.
    pub(super) fn sync(&mut self, config: &ScribaConfig) {
        self.entitlements = config.cloud.entitlements.clone();
        self.requested_at = config.cloud.beta_requested_at.clone();
        if let Some(email) = &config.cloud.email {
            self.email_hint = email.clone();
        }
    }

    pub(super) fn handle_key(&mut self, key: KeyCode) -> CloudAction {
        let phase = std::mem::replace(&mut self.phase, CloudPhase::Menu);
        let (next, action) = match (phase, key) {
            (CloudPhase::Unavailable, _) => (CloudPhase::Unavailable, CloudAction::Finished),

            (CloudPhase::Menu, KeyCode::Char('r') | KeyCode::Char('R')) => (
                CloudPhase::Email {
                    purpose: Purpose::Request,
                    buffer: self.email_hint.clone(),
                },
                CloudAction::Continue,
            ),
            (CloudPhase::Menu, KeyCode::Char('s') | KeyCode::Char('S')) => (
                CloudPhase::Email {
                    purpose: Purpose::SignIn,
                    buffer: self.email_hint.clone(),
                },
                CloudAction::Continue,
            ),
            (CloudPhase::Menu, KeyCode::Esc | KeyCode::Enter) => {
                (CloudPhase::Menu, CloudAction::Finished)
            }
            (CloudPhase::Menu, _) => (CloudPhase::Menu, CloudAction::Continue),

            (
                CloudPhase::Email {
                    purpose,
                    mut buffer,
                },
                key,
            ) => match key {
                KeyCode::Esc => (CloudPhase::Menu, CloudAction::Continue),
                KeyCode::Char(c) if !c.is_whitespace() && buffer.len() < 254 => {
                    buffer.push(c);
                    (CloudPhase::Email { purpose, buffer }, CloudAction::Continue)
                }
                KeyCode::Backspace => {
                    buffer.pop();
                    (CloudPhase::Email { purpose, buffer }, CloudAction::Continue)
                }
                KeyCode::Enter => {
                    if !cloud::looks_like_email(&buffer) {
                        (
                            CloudPhase::Failed {
                                message: "that does not look like an email address".into(),
                                back: Box::new(CloudPhase::Email { purpose, buffer }),
                            },
                            CloudAction::Continue,
                        )
                    } else {
                        let email = cloud::normalize_email(&buffer);
                        self.email_hint = email.clone();
                        match purpose {
                            Purpose::Request => (
                                CloudPhase::Note {
                                    email,
                                    buffer: String::new(),
                                },
                                CloudAction::Continue,
                            ),
                            Purpose::SignIn => (self.start_send_code(email), CloudAction::Continue),
                        }
                    }
                }
                _ => (CloudPhase::Email { purpose, buffer }, CloudAction::Continue),
            },

            (CloudPhase::Note { email, mut buffer }, key) => match key {
                KeyCode::Esc => (
                    CloudPhase::Email {
                        purpose: Purpose::Request,
                        buffer: email,
                    },
                    CloudAction::Continue,
                ),
                KeyCode::Char(c) if buffer.chars().count() < NOTE_MAX => {
                    buffer.push(c);
                    (CloudPhase::Note { email, buffer }, CloudAction::Continue)
                }
                KeyCode::Backspace => {
                    buffer.pop();
                    (CloudPhase::Note { email, buffer }, CloudAction::Continue)
                }
                KeyCode::Enter => (self.start_request(email, buffer), CloudAction::Continue),
                _ => (CloudPhase::Note { email, buffer }, CloudAction::Continue),
            },

            (CloudPhase::Code { email, mut buffer }, key) => match key {
                KeyCode::Esc => (CloudPhase::Menu, CloudAction::Continue),
                KeyCode::Char(c) if c.is_ascii_digit() && buffer.len() < CODE_MAX => {
                    buffer.push(c);
                    (CloudPhase::Code { email, buffer }, CloudAction::Continue)
                }
                KeyCode::Backspace => {
                    buffer.pop();
                    (CloudPhase::Code { email, buffer }, CloudAction::Continue)
                }
                KeyCode::Enter if buffer.len() >= CODE_MIN => {
                    (self.start_verify(email, buffer), CloudAction::Continue)
                }
                _ => (CloudPhase::Code { email, buffer }, CloudAction::Continue),
            },

            // Nothing interrupts a request in flight; it is short.
            (busy @ CloudPhase::Busy { .. }, _) => (busy, CloudAction::Continue),

            (CloudPhase::Requested { .. }, KeyCode::Esc | KeyCode::Enter) => {
                (CloudPhase::Menu, CloudAction::Finished)
            }
            (requested @ CloudPhase::Requested { .. }, _) => (requested, CloudAction::Continue),

            (CloudPhase::Account, KeyCode::Char('r') | KeyCode::Char('R')) => {
                (self.start_refresh(), CloudAction::Continue)
            }
            (CloudPhase::Account, KeyCode::Char('o') | KeyCode::Char('O')) => {
                (self.start_sign_out(), CloudAction::Continue)
            }
            (CloudPhase::Account, KeyCode::Esc | KeyCode::Enter) => {
                (CloudPhase::Account, CloudAction::Finished)
            }
            (CloudPhase::Account, _) => (CloudPhase::Account, CloudAction::Continue),

            (CloudPhase::Failed { back, .. }, KeyCode::Enter) => (*back, CloudAction::Continue),
            (CloudPhase::Failed { .. }, KeyCode::Esc) => (CloudPhase::Menu, CloudAction::Finished),
            (failed @ CloudPhase::Failed { .. }, _) => (failed, CloudAction::Continue),
        };
        self.phase = next;
        action
    }

    fn client(&self) -> SupabaseClient {
        // `new` only leaves `Unavailable` when there is no client, and
        // `Unavailable` never starts a task.
        self.client
            .clone()
            .expect("cloud flow started without a client")
    }

    fn start_request(&self, email: String, note: String) -> CloudPhase {
        let client = self.client();
        let (e, n) = (email.clone(), note);
        CloudPhase::Busy {
            op: Op::Request { email },
            task: tokio::spawn(async move { cloud::request_beta(&client, &e, Some(&n)).await }),
        }
    }

    fn start_send_code(&self, email: String) -> CloudPhase {
        let client = self.client();
        let e = email.clone();
        CloudPhase::Busy {
            op: Op::SendCode { email },
            task: tokio::spawn(async move { cloud::send_code(&client, &e).await }),
        }
    }

    fn start_verify(&self, email: String, code: String) -> CloudPhase {
        let client = self.client();
        let e = email.clone();
        CloudPhase::Busy {
            op: Op::Verify { email },
            task: tokio::spawn(async move { cloud::verify_code(&client, &e, &code).await }),
        }
    }

    fn start_refresh(&self) -> CloudPhase {
        let client = self.client();
        CloudPhase::Busy {
            op: Op::Refresh,
            task: tokio::spawn(async move { cloud::refresh_entitlements(&client).await }),
        }
    }

    fn start_sign_out(&self) -> CloudPhase {
        let client = self.client();
        CloudPhase::Busy {
            op: Op::SignOut,
            task: tokio::spawn(async move { cloud::sign_out(&client).await }),
        }
    }

    /// Collect a finished task. Returns the event for the caller to apply to
    /// the config; the flow already moved to the matching phase.
    pub(super) fn tick(&mut self) -> Option<AccountEvent> {
        let CloudPhase::Busy { task, .. } = &self.phase else {
            return None;
        };
        if !task.is_finished() {
            return None;
        }
        let CloudPhase::Busy { op, task } = std::mem::replace(&mut self.phase, CloudPhase::Menu)
        else {
            unreachable!()
        };
        // Finished, so a single poll yields the result.
        let result = match task.now_or_never() {
            Some(Ok(result)) => result,
            Some(Err(e)) => Err(CloudError::Other(format!("task failed: {e}"))),
            None => Err(CloudError::Other("task did not finish".into())),
        };
        match result {
            Ok(event) => {
                self.phase = self.phase_after(&event);
                Some(event)
            }
            Err(err) => {
                self.phase = self.phase_after_error(op, err);
                None
            }
        }
    }

    fn phase_after(&mut self, event: &AccountEvent) -> CloudPhase {
        match event {
            AccountEvent::BetaRequested { email } => CloudPhase::Requested {
                email: email.clone(),
            },
            AccountEvent::CodeSent { email } => CloudPhase::Code {
                email: email.clone(),
                buffer: String::new(),
            },
            AccountEvent::SignedIn {
                email,
                entitlements,
                ..
            } => {
                self.email_hint = email.clone();
                self.entitlements = entitlements.clone();
                CloudPhase::Account
            }
            AccountEvent::Refreshed { entitlements } => {
                self.entitlements = entitlements.clone();
                CloudPhase::Account
            }
            AccountEvent::SessionLost => CloudPhase::Failed {
                message: "your session expired, sign in again".into(),
                back: Box::new(CloudPhase::Menu),
            },
            AccountEvent::SignedOut => CloudPhase::Menu,
        }
    }

    fn phase_after_error(&self, op: Op, err: CloudError) -> CloudPhase {
        let (message, back) = match (op, &err) {
            (Op::SendCode { .. }, CloudError::NotApproved) => (
                "this email is not on the beta yet \u{00B7} [R] on the previous screen asks to join"
                    .to_string(),
                CloudPhase::Menu,
            ),
            (Op::Verify { email }, CloudError::InvalidCode) => (
                err.to_string(),
                CloudPhase::Code {
                    email,
                    buffer: String::new(),
                },
            ),
            (Op::Verify { email }, _) => (
                err.to_string(),
                CloudPhase::Code {
                    email,
                    buffer: String::new(),
                },
            ),
            (Op::Request { email }, CloudError::Rejected(_)) => (
                err.to_string(),
                CloudPhase::Email {
                    purpose: Purpose::Request,
                    buffer: email,
                },
            ),
            (Op::Request { email }, _) => (
                err.to_string(),
                CloudPhase::Email {
                    purpose: Purpose::Request,
                    buffer: email,
                },
            ),
            (Op::SendCode { email }, _) => (
                err.to_string(),
                CloudPhase::Email {
                    purpose: Purpose::SignIn,
                    buffer: email,
                },
            ),
            (Op::Refresh, _) | (Op::SignOut, _) => (err.to_string(), CloudPhase::Account),
        };
        CloudPhase::Failed {
            message,
            back: Box::new(back),
        }
    }

    /// Key hint for the current phase.
    pub(super) fn hint(&self) -> &'static str {
        match &self.phase {
            CloudPhase::Unavailable => "[Enter] Close",
            CloudPhase::Menu => "[R] Request access  [S] Sign in  [Esc] Close",
            CloudPhase::Email { .. } => "[Enter] Continue  [Esc] Back",
            CloudPhase::Note { .. } => "[Enter] Send  [Esc] Back",
            CloudPhase::Code { .. } => "[Enter] Verify  [Esc] Cancel",
            CloudPhase::Busy { .. } => "",
            CloudPhase::Requested { .. } => "[Enter] Close",
            CloudPhase::Account => "[R] Refresh  [O] Sign out  [Esc] Close",
            CloudPhase::Failed { .. } => "[Enter] Back  [Esc] Close",
        }
    }

    pub(super) fn render_lines(&self, width: usize) -> Vec<Line<'static>> {
        let dim = Style::default().fg(Color::DarkGray);
        let white = Style::default().fg(Color::White);
        let accent = Style::default().fg(ACCENT);
        let bold = Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::BOLD);
        let editing = Style::default()
            .fg(Color::Green)
            .add_modifier(Modifier::BOLD);
        let bad = Style::default().fg(Color::Red);
        let good = Style::default().fg(Color::Green);
        let wrap = |text: &str, style: Style| -> Vec<Line<'static>> {
            textwrap::wrap(text, width.max(20))
                .into_iter()
                .map(|l| Line::from(Span::styled(l.into_owned(), style)))
                .collect()
        };

        let mut lines: Vec<Line<'static>> = Vec::new();
        match &self.phase {
            CloudPhase::Unavailable => {
                lines.extend(wrap(
                    "Scriba Pro is not configured in this build. Set SCRIBA_SUPABASE_URL and SCRIBA_SUPABASE_ANON_KEY to point a development build at a project.",
                    dim,
                ));
            }
            CloudPhase::Menu => {
                lines.extend(wrap(
                    "Scriba Pro is a hosted tier in closed beta: an account, models without API keys, backups, and calendar integration. The open source version keeps working on its own.",
                    dim,
                ));
                lines.push(Line::from(""));
                match &self.requested_at {
                    Some(at) => {
                        lines.push(Line::from(vec![
                            Span::styled("   \u{25CB} ", accent),
                            Span::styled("Access requested ", white),
                            Span::styled(format!("on {}", short_date(at)), dim),
                            Span::styled(" \u{00B7} you will get an email if approved", dim),
                        ]));
                    }
                    None => {
                        lines.push(Line::from(vec![
                            Span::styled("   [R] ", accent),
                            Span::styled("Request access", bold),
                            Span::styled("  ask to join the beta", dim),
                        ]));
                    }
                }
                lines.push(Line::from(vec![
                    Span::styled("   [S] ", accent),
                    Span::styled("Sign in", bold),
                    Span::styled("  with the email that was approved", dim),
                ]));
            }
            CloudPhase::Email { purpose, buffer } => {
                let title = match purpose {
                    Purpose::Request => "Which email should we approve?",
                    Purpose::SignIn => "Email of your Scriba Pro account",
                };
                lines.push(Line::from(Span::styled(title, white)));
                lines.push(Line::from(vec![
                    Span::styled("   email  ", dim),
                    Span::styled(format!("{buffer}_"), editing),
                ]));
            }
            CloudPhase::Note { email, buffer } => {
                lines.push(Line::from(vec![
                    Span::styled("Requesting access for ", white),
                    Span::styled(email.clone(), accent),
                ]));
                lines.push(Line::from(Span::styled(
                    "   A line about how you would use Scriba helps (optional).",
                    dim,
                )));
                lines.push(Line::from(vec![
                    Span::styled("   note   ", dim),
                    Span::styled(format!("{buffer}_"), editing),
                ]));
            }
            CloudPhase::Code { email, buffer } => {
                lines.push(Line::from(vec![
                    Span::styled("We emailed a code to ", white),
                    Span::styled(email.clone(), accent),
                ]));
                lines.push(Line::from(vec![
                    Span::styled("   code   ", dim),
                    Span::styled(format!("{buffer}_"), editing),
                ]));
            }
            CloudPhase::Busy { op, .. } => {
                let what = match op {
                    Op::Request { .. } => "Sending your request\u{2026}",
                    Op::SendCode { .. } => "Emailing you a code\u{2026}",
                    Op::Verify { .. } => "Checking the code\u{2026}",
                    Op::Refresh => "Refreshing\u{2026}",
                    Op::SignOut => "Signing out\u{2026}",
                };
                lines.push(Line::from(Span::styled(what, accent)));
            }
            CloudPhase::Requested { email } => {
                lines.push(Line::from(vec![
                    Span::styled("\u{2713} ", good),
                    Span::styled("Request sent for ", white),
                    Span::styled(email.clone(), accent),
                ]));
                lines.extend(wrap(
                    "The beta is invite-only and approved by hand. You will get an email when your account is ready; then come back here and sign in.",
                    dim,
                ));
            }
            CloudPhase::Account => {
                lines.push(Line::from(vec![
                    Span::styled("Signed in as ", white),
                    Span::styled(self.email_hint.clone(), accent),
                ]));
                if self.entitlements.is_empty() {
                    lines.push(Line::from(vec![
                        Span::styled("   \u{25CB} ", dim),
                        Span::styled("no features unlocked yet", dim),
                    ]));
                } else {
                    for feature in &self.entitlements {
                        lines.push(Line::from(vec![
                            Span::styled("   \u{2713} ", good),
                            Span::styled(feature_label(feature), white),
                        ]));
                    }
                }
            }
            CloudPhase::Failed { message, .. } => {
                lines.extend(wrap(&format!("\u{2717} {message}"), bad));
            }
        }
        if !self.hint().is_empty() {
            lines.push(Line::from(Span::styled(self.hint(), dim)));
        }
        lines
    }
}

impl std::fmt::Debug for CloudFlow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let phase = match &self.phase {
            CloudPhase::Unavailable => "Unavailable",
            CloudPhase::Menu => "Menu",
            CloudPhase::Email { .. } => "Email",
            CloudPhase::Note { .. } => "Note",
            CloudPhase::Code { .. } => "Code",
            CloudPhase::Busy { .. } => "Busy",
            CloudPhase::Requested { .. } => "Requested",
            CloudPhase::Account => "Account",
            CloudPhase::Failed { .. } => "Failed",
        };
        f.debug_struct("CloudFlow").field("phase", &phase).finish()
    }
}

/// Human name of an entitlement.
pub(super) fn feature_label(feature: &str) -> String {
    match feature {
        "beta" => "Beta member".to_string(),
        "pro" => "Pro".to_string(),
        other => other.to_string(),
    }
}

/// "2026-09-28T10:00:00+00:00" -> "2026-09-28".
fn short_date(rfc3339: &str) -> String {
    rfc3339.split('T').next().unwrap_or(rfc3339).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn configured() -> ScribaConfig {
        let mut config = ScribaConfig::default();
        config.cloud.supabase_url = Some("https://x.supabase.co".into());
        config.cloud.supabase_anon_key = Some("anon".into());
        config
    }

    fn type_str(flow: &mut CloudFlow, s: &str) {
        for c in s.chars() {
            flow.handle_key(KeyCode::Char(c));
        }
    }

    #[test]
    fn unconfigured_build_closes_on_any_key() {
        let mut flow = CloudFlow::new(&ScribaConfig::default());
        if std::env::var("SCRIBA_SUPABASE_URL").is_ok() || !cloud::SUPABASE_URL.is_empty() {
            return;
        }
        assert!(matches!(flow.phase, CloudPhase::Unavailable));
        assert_eq!(flow.handle_key(KeyCode::Char('r')), CloudAction::Finished);
    }

    #[test]
    fn request_path_validates_the_email_before_asking_for_a_note() {
        let mut flow = CloudFlow::new(&configured());
        assert!(matches!(flow.phase, CloudPhase::Menu));
        flow.handle_key(KeyCode::Char('r'));
        type_str(&mut flow, "not an email");
        assert!(matches!(&flow.phase, CloudPhase::Email { buffer, .. } if buffer == "notanemail"));
        flow.handle_key(KeyCode::Enter);
        assert!(matches!(flow.phase, CloudPhase::Failed { .. }));
        flow.handle_key(KeyCode::Enter); // back to the email field
        for _ in 0..20 {
            flow.handle_key(KeyCode::Backspace);
        }
        type_str(&mut flow, "Me@Example.com");
        flow.handle_key(KeyCode::Enter);
        assert!(matches!(&flow.phase, CloudPhase::Note { email, .. } if email == "me@example.com"));
        assert_eq!(flow.hint(), "[Enter] Send  [Esc] Back");
    }

    #[test]
    fn code_entry_accepts_digits_only_and_needs_six() {
        let mut flow = CloudFlow::new(&configured());
        flow.phase = CloudPhase::Code {
            email: "me@example.com".into(),
            buffer: String::new(),
        };
        type_str(&mut flow, "12a34");
        assert!(matches!(&flow.phase, CloudPhase::Code { buffer, .. } if buffer == "1234"));
        flow.handle_key(KeyCode::Enter);
        assert!(
            matches!(&flow.phase, CloudPhase::Code { .. }),
            "too short to submit"
        );
        assert_eq!(flow.handle_key(KeyCode::Esc), CloudAction::Continue);
        assert!(matches!(flow.phase, CloudPhase::Menu));
    }

    #[test]
    fn events_move_the_flow_forward() {
        let mut flow = CloudFlow::new(&configured());
        let phase = flow.phase_after(&AccountEvent::CodeSent {
            email: "me@example.com".into(),
        });
        assert!(matches!(phase, CloudPhase::Code { .. }));
        let phase = flow.phase_after(&AccountEvent::SignedIn {
            email: "me@example.com".into(),
            user_id: "u".into(),
            entitlements: vec!["beta".into()],
        });
        assert!(matches!(phase, CloudPhase::Account));
        assert_eq!(flow.entitlements, vec!["beta".to_string()]);
        let phase = flow.phase_after(&AccountEvent::SessionLost);
        assert!(matches!(phase, CloudPhase::Failed { .. }));
    }

    #[test]
    fn errors_return_to_the_right_screen() {
        let flow = CloudFlow::new(&configured());
        let phase = flow.phase_after_error(
            Op::SendCode {
                email: "a@b.co".into(),
            },
            CloudError::NotApproved,
        );
        let CloudPhase::Failed { back, message } = phase else {
            panic!("expected failure");
        };
        assert!(message.contains("not on the beta"));
        assert!(matches!(*back, CloudPhase::Menu));

        let phase = flow.phase_after_error(
            Op::Verify {
                email: "a@b.co".into(),
            },
            CloudError::InvalidCode,
        );
        let CloudPhase::Failed { back, .. } = phase else {
            panic!("expected failure");
        };
        assert!(matches!(*back, CloudPhase::Code { .. }));
    }

    #[test]
    fn rendering_never_panics_on_narrow_widths() {
        let mut flow = CloudFlow::new(&configured());
        for phase in [
            CloudPhase::Unavailable,
            CloudPhase::Menu,
            CloudPhase::Requested {
                email: "a@b.co".into(),
            },
            CloudPhase::Account,
            CloudPhase::Failed {
                message: "x".repeat(200),
                back: Box::new(CloudPhase::Menu),
            },
        ] {
            flow.phase = phase;
            for width in [1, 10, 40, 120] {
                assert!(!flow.render_lines(width).is_empty());
            }
        }
    }
}
