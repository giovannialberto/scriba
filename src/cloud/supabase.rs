//! Thin client for the Supabase project behind Scriba Pro.
//!
//! Only two surfaces are used: GoTrue for email one-time-code sign-in and
//! PostgREST for the beta request and entitlement tables. The anon key is a
//! public identifier of the project, not a secret; row level security on the
//! server decides what it can do (insert a beta request, nothing else).

use std::fmt;
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// A signed-in session. The access token is short-lived and kept in memory;
/// the refresh token is what gets stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Session {
    pub access_token: String,
    pub refresh_token: String,
    /// Unix timestamp when `access_token` expires.
    pub expires_at: i64,
    pub user_id: String,
    pub email: String,
}

/// A feature the account holds.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct Entitlement {
    pub feature: String,
    #[serde(default)]
    pub expires_at: Option<String>,
}

/// This month's proxy usage for the signed-in account.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq, Default)]
pub struct UsageSummary {
    #[serde(default)]
    pub requests: i64,
    #[serde(default)]
    pub input_tokens: i64,
    #[serde(default)]
    pub output_tokens: i64,
    #[serde(default)]
    pub response_bytes: i64,
    /// Seconds of audio transcribed this month.
    #[serde(default)]
    pub audio_seconds: i64,
}

impl UsageSummary {
    pub fn tokens(&self) -> i64 {
        self.input_tokens + self.output_tokens
    }

    /// "45 min", "2.5 h", or "" when nothing was transcribed.
    pub fn audio_display(&self) -> String {
        match self.audio_seconds {
            s if s <= 0 => String::new(),
            s if s < 3600 => format!("{} min", (s + 59) / 60),
            s => format!("{:.1} h", s as f64 / 3600.0),
        }
    }
}

/// Why a call failed, in terms the UI can explain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CloudError {
    /// The email has no account: the beta request was not approved (yet).
    NotApproved,
    /// The code was wrong or has expired.
    InvalidCode,
    /// The stored session is no longer valid: sign in again.
    SessionExpired,
    /// Too many attempts; try again later.
    RateLimited,
    /// The request was accepted but the server rejected the data
    /// (bad email, note too long, duplicate).
    Rejected(String),
    /// Could not reach the server.
    Network(String),
    /// Anything else, with the server's message.
    Other(String),
}

impl fmt::Display for CloudError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CloudError::NotApproved => {
                write!(f, "this email is not on the beta yet")
            }
            CloudError::InvalidCode => write!(f, "wrong or expired code"),
            CloudError::SessionExpired => write!(f, "session expired, sign in again"),
            CloudError::RateLimited => write!(f, "too many attempts, wait a minute"),
            CloudError::Rejected(msg) => write!(f, "{msg}"),
            CloudError::Network(msg) => write!(f, "cannot reach Scriba Pro: {msg}"),
            CloudError::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for CloudError {}

/// Map an HTTP status and body to a [`CloudError`]. Pure, so it can be tested
/// without a server. GoTrue answers with either `{"error_code","msg"}` (new)
/// or `{"error","error_description"}` / `{"msg"}` (old); PostgREST with
/// `{"code","message","details"}`.
pub fn classify(status: u16, body: &str) -> CloudError {
    let json: serde_json::Value = serde_json::from_str(body).unwrap_or_default();
    let field = |k: &str| {
        json.get(k)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string()
    };
    let code = field("error_code");
    let msg = {
        let m = field("msg");
        if !m.is_empty() {
            m
        } else {
            let m = field("error_description");
            if !m.is_empty() {
                m
            } else {
                let m = field("message");
                if !m.is_empty() { m } else { field("error") }
            }
        }
    };
    let lower = msg.to_lowercase();

    if status == 429 || code == "over_email_send_rate_limit" || lower.contains("rate limit") {
        return CloudError::RateLimited;
    }
    // PostgREST: unique violation on the pending-request index.
    if status == 409 || field("code") == "23505" {
        return CloudError::Rejected("this email already has a pending request".into());
    }
    if code == "otp_disabled"
        || code == "signup_disabled"
        || lower.contains("signups not allowed")
        || lower.contains("signup disabled")
    {
        return CloudError::NotApproved;
    }
    if code == "otp_expired"
        || lower.contains("expired or is invalid")
        || lower.contains("invalid otp")
    {
        return CloudError::InvalidCode;
    }
    if code == "refresh_token_not_found"
        || code == "refresh_token_already_used"
        || code == "session_not_found"
        || code == "bad_jwt"
        || lower.contains("invalid refresh token")
        || lower.contains("refresh token not found")
        || lower.contains("jwt expired")
    {
        return CloudError::SessionExpired;
    }
    if status == 401 || status == 403 {
        return CloudError::SessionExpired;
    }
    if (400..500).contains(&status) {
        let text = if msg.is_empty() {
            format!("request rejected ({status})")
        } else {
            msg
        };
        return CloudError::Rejected(text);
    }
    CloudError::Other(if msg.is_empty() {
        format!("server error ({status})")
    } else {
        format!("{msg} ({status})")
    })
}

#[derive(Deserialize)]
struct SessionResponse {
    access_token: String,
    refresh_token: String,
    #[serde(default)]
    expires_at: Option<i64>,
    #[serde(default)]
    expires_in: Option<i64>,
    user: UserResponse,
}

#[derive(Deserialize)]
struct UserResponse {
    id: String,
    #[serde(default)]
    email: String,
}

impl SessionResponse {
    fn into_session(self) -> Session {
        let now = chrono::Utc::now().timestamp();
        let expires_at = self
            .expires_at
            .or_else(|| self.expires_in.map(|s| now + s))
            .unwrap_or(now + 3600);
        Session {
            access_token: self.access_token,
            refresh_token: self.refresh_token,
            expires_at,
            user_id: self.user.id,
            email: self.user.email,
        }
    }
}

#[derive(Serialize)]
struct BetaRequestBody<'a> {
    email: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    note: Option<&'a str>,
    client_version: &'a str,
}

/// Client for one Supabase project.
#[derive(Debug, Clone)]
pub struct SupabaseClient {
    url: String,
    anon_key: String,
    http: reqwest::Client,
}

impl SupabaseClient {
    pub fn new(url: &str, anon_key: &str) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(20))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self {
            url: url.trim_end_matches('/').to_string(),
            anon_key: anon_key.to_string(),
            http,
        }
    }

    /// Project URL, for display.
    pub fn url(&self) -> &str {
        &self.url
    }

    fn request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        self.http
            .request(method, format!("{}{}", self.url, path))
            .header("apikey", &self.anon_key)
    }

    async fn send(&self, req: reqwest::RequestBuilder) -> Result<String, CloudError> {
        let resp = req
            .send()
            .await
            .map_err(|e| CloudError::Network(compact_reqwest_error(&e)))?;
        let status = resp.status().as_u16();
        let body = resp.text().await.unwrap_or_default();
        if (200..300).contains(&status) {
            Ok(body)
        } else {
            Err(classify(status, &body))
        }
    }

    /// Email a one-time code to an existing (approved) account.
    pub async fn send_code(&self, email: &str) -> Result<(), CloudError> {
        let body = serde_json::json!({ "email": email, "create_user": false });
        self.send(
            self.request(reqwest::Method::POST, "/auth/v1/otp")
                .json(&body),
        )
        .await
        .map(|_| ())
    }

    /// Exchange the emailed code for a session.
    pub async fn verify_code(&self, email: &str, code: &str) -> Result<Session, CloudError> {
        let body = serde_json::json!({ "type": "email", "email": email, "token": code });
        let text = self
            .send(
                self.request(reqwest::Method::POST, "/auth/v1/verify")
                    .json(&body),
            )
            .await?;
        parse_session(&text)
    }

    /// Get a fresh session from a stored refresh token. Supabase rotates the
    /// refresh token: store the returned one.
    pub async fn refresh(&self, refresh_token: &str) -> Result<Session, CloudError> {
        let body = serde_json::json!({ "refresh_token": refresh_token });
        let text = self
            .send(
                self.request(
                    reqwest::Method::POST,
                    "/auth/v1/token?grant_type=refresh_token",
                )
                .json(&body),
            )
            .await?;
        parse_session(&text)
    }

    /// Revoke the session server-side. Best effort: callers forget the local
    /// token regardless.
    pub async fn sign_out(&self, access_token: &str) -> Result<(), CloudError> {
        self.send(
            self.request(reqwest::Method::POST, "/auth/v1/logout?scope=global")
                .bearer_auth(access_token),
        )
        .await
        .map(|_| ())
    }

    /// Ask to join the closed beta.
    pub async fn request_beta(
        &self,
        email: &str,
        note: Option<&str>,
        client_version: &str,
    ) -> Result<(), CloudError> {
        let body = BetaRequestBody {
            email,
            note: note.map(str::trim).filter(|n| !n.is_empty()),
            client_version,
        };
        self.send(
            self.request(reqwest::Method::POST, "/rest/v1/beta_requests")
                .header("Prefer", "return=minimal")
                .json(&body),
        )
        .await
        .map(|_| ())
    }

    /// This month's usage, from the `usage_this_month` function.
    pub async fn usage_this_month(
        &self,
        access_token: &str,
        user_id: &str,
    ) -> Result<UsageSummary, CloudError> {
        let body = serde_json::json!({ "uid": user_id });
        let text = self
            .send(
                self.request(reqwest::Method::POST, "/rest/v1/rpc/usage_this_month")
                    .bearer_auth(access_token)
                    .json(&body),
            )
            .await?;
        // A table-returning function comes back as an array of rows.
        let rows: Vec<UsageSummary> = serde_json::from_str(&text)
            .or_else(|_| serde_json::from_str::<UsageSummary>(&text).map(|u| vec![u]))
            .map_err(|e| CloudError::Other(format!("bad usage response: {e}")))?;
        Ok(rows.into_iter().next().unwrap_or_default())
    }

    /// Features the signed-in account holds.
    pub async fn entitlements(&self, access_token: &str) -> Result<Vec<Entitlement>, CloudError> {
        let text = self
            .send(
                self.request(
                    reqwest::Method::GET,
                    "/rest/v1/entitlements?select=feature,expires_at",
                )
                .bearer_auth(access_token),
            )
            .await?;
        let all: Vec<Entitlement> = serde_json::from_str(&text)
            .map_err(|e| CloudError::Other(format!("bad entitlements: {e}")))?;
        Ok(all.into_iter().filter(|e| !expired(e)).collect())
    }
}

fn parse_session(text: &str) -> Result<Session, CloudError> {
    serde_json::from_str::<SessionResponse>(text)
        .map(SessionResponse::into_session)
        .map_err(|e| CloudError::Other(format!("bad session response: {e}")))
}

fn expired(e: &Entitlement) -> bool {
    match &e.expires_at {
        None => false,
        Some(ts) => chrono::DateTime::parse_from_rfc3339(ts)
            .map(|t| t < chrono::Utc::now())
            .unwrap_or(false),
    }
}

fn compact_reqwest_error(e: &reqwest::Error) -> String {
    if e.is_timeout() {
        "timed out".to_string()
    } else if e.is_connect() {
        "connection failed".to_string()
    } else {
        let mut s = e.to_string();
        if let Some(idx) = s.find(": ") {
            s = s[idx + 2..].to_string();
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unapproved_emails_are_recognized() {
        let body =
            r#"{"code":422,"error_code":"otp_disabled","msg":"Signups not allowed for otp"}"#;
        assert_eq!(classify(422, body), CloudError::NotApproved);
        let legacy = r#"{"error":"Signups not allowed for otp"}"#;
        assert_eq!(classify(400, legacy), CloudError::NotApproved);
    }

    #[test]
    fn bad_codes_and_dead_sessions_are_recognized() {
        let body =
            r#"{"code":403,"error_code":"otp_expired","msg":"Token has expired or is invalid"}"#;
        assert_eq!(classify(403, body), CloudError::InvalidCode);
        let body = r#"{"error_code":"refresh_token_not_found","msg":"Invalid Refresh Token: Refresh Token Not Found"}"#;
        assert_eq!(classify(400, body), CloudError::SessionExpired);
        assert_eq!(classify(401, "{}"), CloudError::SessionExpired);
    }

    #[test]
    fn rate_limits_and_rejections_keep_their_message() {
        assert_eq!(classify(429, ""), CloudError::RateLimited);
        let body =
            r#"{"error_code":"over_email_send_rate_limit","msg":"email rate limit exceeded"}"#;
        assert_eq!(classify(400, body), CloudError::RateLimited);
        let body =
            r#"{"code":"23514","message":"new row violates check constraint","details":null}"#;
        assert_eq!(
            classify(400, body),
            CloudError::Rejected("new row violates check constraint".into())
        );
        assert!(matches!(classify(500, "oops"), CloudError::Other(_)));
        let dup = r#"{"code":"23505","message":"duplicate key value violates unique constraint"}"#;
        assert_eq!(
            classify(409, dup),
            CloudError::Rejected("this email already has a pending request".into())
        );
    }

    #[test]
    fn sessions_parse_with_either_expiry_field() {
        let text = r#"{"access_token":"a","refresh_token":"r","expires_in":3600,"user":{"id":"u1","email":"me@x.io"}}"#;
        let s = parse_session(text).unwrap();
        assert_eq!(s.user_id, "u1");
        assert_eq!(s.email, "me@x.io");
        assert!(s.expires_at > chrono::Utc::now().timestamp() + 3000);
        let text = r#"{"access_token":"a","refresh_token":"r","expires_at":42,"user":{"id":"u1"}}"#;
        assert_eq!(parse_session(text).unwrap().expires_at, 42);
        assert!(parse_session("{}").is_err());
    }

    #[test]
    fn audio_usage_reads_naturally() {
        let mut u = UsageSummary::default();
        assert_eq!(u.audio_display(), "");
        u.audio_seconds = 61;
        assert_eq!(u.audio_display(), "2 min");
        u.audio_seconds = 9000;
        assert_eq!(u.audio_display(), "2.5 h");
        let parsed: UsageSummary = serde_json::from_str(r#"{"requests":1,"unmetered_bytes":5}"#).unwrap();
        assert_eq!(parsed.requests, 1, "unknown fields are ignored");
    }

    #[test]
    fn expired_entitlements_are_dropped() {
        let live = Entitlement {
            feature: "beta".into(),
            expires_at: None,
        };
        let future = Entitlement {
            feature: "pro".into(),
            expires_at: Some("2999-01-01T00:00:00Z".into()),
        };
        let past = Entitlement {
            feature: "old".into(),
            expires_at: Some("2001-01-01T00:00:00Z".into()),
        };
        assert!(!expired(&live));
        assert!(!expired(&future));
        assert!(expired(&past));
    }
}
