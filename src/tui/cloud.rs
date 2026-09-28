//! Scriba Pro flows, shown inline under a Settings row.
//!
//! Two separate, linear flows: sign in (email, then the emailed code) and
//! request access (email, then an optional note). Enter moves forward, Esc
//! moves back, errors appear under the field, and the panel closes itself
//! when the flow is done. Signed-in actions (refresh, sign out) are a normal
//! Settings picker, not part of this component.

use crossterm::event::KeyCode;
use futures_util::FutureExt;
use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
};
use tokio::task::JoinHandle;

use super::chat::ACCENT;
use crate::cloud::{self, AccountEvent, CloudError, SupabaseClient};
use crate::core::ScribaConfig;

/// Longest note accepted with a beta request (matches the server).
const NOTE_MAX: usize = 500;
/// Codes are eight digits today; accept a little either way.
const CODE_MAX: usize = 10;
const CODE_MIN: usize = 6;

/// Which flow is running.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Goal {
    SignIn,
    RequestAccess,
}

/// What a background task is doing, so a failure knows which field to
/// return to.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Op {
    Request { email: String },
    SendCode { email: String },
    Verify { email: String },
}

enum Step {
    Email,
    Note {
        email: String,
    },
    Code {
        email: String,
    },
    Busy {
        op: Op,
        task: JoinHandle<Result<AccountEvent, CloudError>>,
    },
    Done,
}

/// Outcome of a key press.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CloudAction {
    Continue,
    /// The flow is over (cancelled or finished): the caller closes it.
    Finished,
}

pub(super) struct CloudFlow {
    goal: Goal,
    step: Step,
    /// Text of the field being edited.
    buffer: String,
    /// Shown in red under the field until the user types again.
    error: Option<String>,
    client: SupabaseClient,
}

impl CloudFlow {
    /// Start a flow. `None` when this build has no project configured, in
    /// which case Settings never offers the rows that would open one.
    pub(super) fn new(goal: Goal, config: &ScribaConfig) -> Option<Self> {
        let client = cloud::client_for(config)?;
        Some(Self {
            goal,
            step: Step::Email,
            buffer: config.cloud.email.clone().unwrap_or_default(),
            error: None,
            client,
        })
    }

    pub(super) fn goal(&self) -> Goal {
        self.goal
    }

    /// Whether the flow completed successfully (the caller closes it).
    pub(super) fn is_done(&self) -> bool {
        matches!(self.step, Step::Done)
    }

    pub(super) fn handle_key(&mut self, key: KeyCode) -> CloudAction {
        match key {
            KeyCode::Esc => self.back(),
            KeyCode::Enter => self.forward(),
            KeyCode::Backspace => {
                if !matches!(self.step, Step::Busy { .. } | Step::Done) {
                    self.buffer.pop();
                    self.error = None;
                }
                CloudAction::Continue
            }
            KeyCode::Char(c) => {
                let accept = match &self.step {
                    Step::Email => !c.is_whitespace() && self.buffer.len() < 254,
                    Step::Note { .. } => self.buffer.chars().count() < NOTE_MAX,
                    Step::Code { .. } => c.is_ascii_digit() && self.buffer.len() < CODE_MAX,
                    Step::Busy { .. } | Step::Done => false,
                };
                if accept {
                    self.buffer.push(c);
                    self.error = None;
                }
                CloudAction::Continue
            }
            _ => CloudAction::Continue,
        }
    }

    /// Esc: one step back, or out of the flow from the first field.
    fn back(&mut self) -> CloudAction {
        let step = std::mem::replace(&mut self.step, Step::Done);
        self.error = None;
        match step {
            Step::Email | Step::Code { .. } | Step::Done => CloudAction::Finished,
            Step::Note { email } => {
                self.buffer = email;
                self.step = Step::Email;
                CloudAction::Continue
            }
            busy @ Step::Busy { .. } => {
                // A request in flight is short; let it finish.
                self.step = busy;
                CloudAction::Continue
            }
        }
    }

    /// Enter: validate the field and move on.
    fn forward(&mut self) -> CloudAction {
        let step = std::mem::replace(&mut self.step, Step::Done);
        match step {
            Step::Email => {
                if !cloud::looks_like_email(&self.buffer) {
                    self.error = Some("that does not look like an email address".into());
                    self.step = Step::Email;
                    return CloudAction::Continue;
                }
                let email = cloud::normalize_email(&self.buffer);
                self.buffer.clear();
                self.step = match self.goal {
                    Goal::SignIn => self.start_send_code(email),
                    Goal::RequestAccess => Step::Note { email },
                };
            }
            Step::Note { email } => {
                let note = std::mem::take(&mut self.buffer);
                self.step = self.start_request(email, note);
            }
            Step::Code { email } => {
                if self.buffer.len() < CODE_MIN {
                    self.error = Some("type the whole code from the email".into());
                    self.step = Step::Code { email };
                    return CloudAction::Continue;
                }
                let code = std::mem::take(&mut self.buffer);
                self.step = self.start_verify(email, code);
            }
            busy @ Step::Busy { .. } => self.step = busy,
            Step::Done => return CloudAction::Finished,
        }
        CloudAction::Continue
    }

    fn start_request(&self, email: String, note: String) -> Step {
        let client = self.client.clone();
        let e = email.clone();
        Step::Busy {
            op: Op::Request { email },
            task: tokio::spawn(async move { cloud::request_beta(&client, &e, Some(&note)).await }),
        }
    }

    fn start_send_code(&self, email: String) -> Step {
        let client = self.client.clone();
        let e = email.clone();
        Step::Busy {
            op: Op::SendCode { email },
            task: tokio::spawn(async move { cloud::send_code(&client, &e).await }),
        }
    }

    fn start_verify(&self, email: String, code: String) -> Step {
        let client = self.client.clone();
        let e = email.clone();
        Step::Busy {
            op: Op::Verify { email },
            task: tokio::spawn(async move { cloud::verify_code(&client, &e, &code).await }),
        }
    }

    /// Collect a finished task. Returns the event for the caller to apply to
    /// the config; the flow already moved to the next step.
    pub(super) fn tick(&mut self) -> Option<AccountEvent> {
        let Step::Busy { task, .. } = &self.step else {
            return None;
        };
        if !task.is_finished() {
            return None;
        }
        let Step::Busy { op, task } = std::mem::replace(&mut self.step, Step::Done) else {
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
                self.step = Self::step_after(&event);
                Some(event)
            }
            Err(err) => {
                self.step_after_error(op, err);
                None
            }
        }
    }

    fn step_after(event: &AccountEvent) -> Step {
        match event {
            AccountEvent::CodeSent { email } => Step::Code {
                email: email.clone(),
            },
            _ => Step::Done,
        }
    }

    /// Go back to the field the failure belongs to, with the message under it.
    fn step_after_error(&mut self, op: Op, err: CloudError) {
        let (step, buffer, message) = match (op, &err) {
            (Op::SendCode { email }, CloudError::NotApproved) => (
                Step::Email,
                email,
                "not on the beta yet \u{00B7} use Request access below".to_string(),
            ),
            (Op::SendCode { email }, _) | (Op::Request { email }, _) => {
                (Step::Email, email, err.to_string())
            }
            (Op::Verify { email }, _) => (Step::Code { email }, String::new(), err.to_string()),
        };
        self.step = step;
        self.buffer = buffer;
        self.error = Some(message);
    }

    pub(super) fn render_lines(&self, width: usize) -> Vec<Line<'static>> {
        let dim = Style::default().fg(Color::DarkGray);
        let white = Style::default().fg(Color::White);
        let accent = Style::default().fg(ACCENT);
        let editing = Style::default()
            .fg(Color::Green)
            .add_modifier(Modifier::BOLD);
        let bad = Style::default().fg(Color::Red);
        let wrap = |text: &str, style: Style| -> Vec<Line<'static>> {
            textwrap::wrap(text, width.max(20))
                .into_iter()
                .map(|l| Line::from(Span::styled(l.into_owned(), style)))
                .collect()
        };

        let mut lines: Vec<Line<'static>> = Vec::new();
        let field = |label: &str, buffer: &str| {
            Line::from(vec![
                Span::styled(format!("{label:<7}"), dim),
                Span::styled(format!("{buffer}_"), editing),
            ])
        };
        match &self.step {
            Step::Email => {
                let title = match self.goal {
                    Goal::SignIn => "Sign in with the email that was approved for the beta.",
                    Goal::RequestAccess => {
                        "Ask to join the closed beta. Approval is by hand; you get an email either way."
                    }
                };
                lines.extend(wrap(title, white));
                lines.push(field("email", &self.buffer));
            }
            Step::Note { email } => {
                lines.push(Line::from(vec![
                    Span::styled("Requesting access for ", white),
                    Span::styled(email.clone(), accent),
                ]));
                lines.push(Line::from(Span::styled(
                    "A line about how you would use Scriba helps. Optional.",
                    dim,
                )));
                lines.push(field("note", &self.buffer));
            }
            Step::Code { email } => {
                lines.push(Line::from(vec![
                    Span::styled("We emailed a code to ", white),
                    Span::styled(email.clone(), accent),
                    Span::styled(". It expires in 10 minutes.", white),
                ]));
                lines.push(field("code", &self.buffer));
            }
            Step::Busy { op, .. } => {
                let what = match op {
                    Op::Request { .. } => "Sending your request\u{2026}",
                    Op::SendCode { .. } => "Emailing you a code\u{2026}",
                    Op::Verify { .. } => "Checking the code\u{2026}",
                };
                lines.push(Line::from(Span::styled(what, accent)));
            }
            Step::Done => {}
        }
        if let Some(error) = &self.error {
            lines.extend(wrap(&format!("\u{2717} {error}"), bad));
        }
        let hint = match &self.step {
            Step::Email | Step::Code { .. } => "Enter to continue \u{00B7} Esc to cancel",
            Step::Note { .. } => "Enter to send \u{00B7} Esc to go back",
            Step::Busy { .. } | Step::Done => "",
        };
        if !hint.is_empty() {
            lines.push(Line::from(Span::styled(hint, dim)));
        }
        lines
    }
}

impl std::fmt::Debug for CloudFlow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let step = match &self.step {
            Step::Email => "Email",
            Step::Note { .. } => "Note",
            Step::Code { .. } => "Code",
            Step::Busy { .. } => "Busy",
            Step::Done => "Done",
        };
        f.debug_struct("CloudFlow")
            .field("goal", &self.goal)
            .field("step", &step)
            .finish()
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
pub(super) fn short_date(rfc3339: &str) -> String {
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
    fn invalid_email_shows_inline_error_and_stays_on_the_field() {
        let mut flow = CloudFlow::new(Goal::RequestAccess, &configured()).unwrap();
        type_str(&mut flow, "not an email");
        assert_eq!(flow.buffer, "notanemail", "spaces are dropped");
        assert_eq!(flow.handle_key(KeyCode::Enter), CloudAction::Continue);
        assert!(matches!(flow.step, Step::Email));
        assert!(flow.error.is_some());
        flow.handle_key(KeyCode::Char('x'));
        assert!(flow.error.is_none(), "typing clears the error");
    }

    #[test]
    fn request_access_goes_email_then_note_and_esc_steps_back() {
        let mut flow = CloudFlow::new(Goal::RequestAccess, &configured()).unwrap();
        type_str(&mut flow, "Me@Example.com");
        flow.handle_key(KeyCode::Enter);
        assert!(matches!(&flow.step, Step::Note { email } if email == "me@example.com"));
        assert!(flow.buffer.is_empty());
        assert_eq!(flow.handle_key(KeyCode::Esc), CloudAction::Continue);
        assert!(matches!(flow.step, Step::Email));
        assert_eq!(
            flow.buffer, "me@example.com",
            "the email comes back for editing"
        );
        assert_eq!(flow.handle_key(KeyCode::Esc), CloudAction::Finished);
    }

    #[test]
    fn code_entry_accepts_digits_only_and_needs_six() {
        let mut flow = CloudFlow::new(Goal::SignIn, &configured()).unwrap();
        flow.step = Step::Code {
            email: "me@example.com".into(),
        };
        type_str(&mut flow, "12a34");
        assert_eq!(flow.buffer, "1234");
        flow.handle_key(KeyCode::Enter);
        assert!(
            matches!(flow.step, Step::Code { .. }),
            "too short to submit"
        );
        assert!(flow.error.is_some());
        assert_eq!(flow.handle_key(KeyCode::Esc), CloudAction::Finished);
    }

    #[test]
    fn events_move_the_flow_forward() {
        assert!(matches!(
            CloudFlow::step_after(&AccountEvent::CodeSent {
                email: "me@example.com".into()
            }),
            Step::Code { .. }
        ));
        assert!(matches!(
            CloudFlow::step_after(&AccountEvent::SignedIn {
                email: "me@example.com".into(),
                user_id: "u".into(),
                entitlements: vec!["beta".into()],
            }),
            Step::Done
        ));
        assert!(matches!(
            CloudFlow::step_after(&AccountEvent::BetaRequested {
                email: "me@example.com".into()
            }),
            Step::Done
        ));
    }

    #[test]
    fn errors_return_to_the_right_field() {
        let mut flow = CloudFlow::new(Goal::SignIn, &configured()).unwrap();
        flow.step_after_error(
            Op::SendCode {
                email: "a@b.co".into(),
            },
            CloudError::NotApproved,
        );
        assert!(matches!(flow.step, Step::Email));
        assert_eq!(flow.buffer, "a@b.co");
        assert!(flow.error.as_deref().unwrap().contains("Request access"));

        flow.step_after_error(
            Op::Verify {
                email: "a@b.co".into(),
            },
            CloudError::InvalidCode,
        );
        assert!(matches!(&flow.step, Step::Code { email } if email == "a@b.co"));
        assert!(flow.buffer.is_empty(), "a wrong code is cleared");
    }

    #[test]
    fn rendering_never_panics_on_narrow_widths() {
        let mut flow = CloudFlow::new(Goal::SignIn, &configured()).unwrap();
        flow.error = Some("x".repeat(200));
        for step in [
            Step::Email,
            Step::Note {
                email: "a@b.co".into(),
            },
            Step::Code {
                email: "a@b.co".into(),
            },
        ] {
            flow.step = step;
            for width in [1, 10, 40, 120] {
                assert!(!flow.render_lines(width).is_empty());
            }
        }
    }

    #[test]
    fn helpers() {
        assert_eq!(short_date("2026-09-28T10:00:00+00:00"), "2026-09-28");
        assert_eq!(feature_label("beta"), "Beta member");
    }
}
