//! Scriba Pro: the hosted tier of Scriba (closed beta).
//!
//! The open source client stays complete on its own; this module only adds
//! an account. Signing in is by email one-time code against a Supabase
//! project, the refresh token lives in a user-only file (see [`secrets`]), and
//! features are unlocked by entitlements the server grants, never by the
//! mere existence of an account. Nothing here is required for recording,
//! transcription, or the assistant to work.
//!
//! The project URL and anon key below identify the Scriba Pro Supabase
//! project. They are public by design; row level security decides what an
//! anonymous caller can do (ask to join the beta, nothing else). Until they
//! are filled in, the account screen reports Pro as not configured. Both can
//! be overridden for development with `SCRIBA_SUPABASE_URL` and
//! `SCRIBA_SUPABASE_ANON_KEY`, or the matching fields in `config.json`.

pub mod secrets;
pub mod supabase;

use std::sync::{Mutex, OnceLock};

use crate::core::ScribaConfig;
pub use supabase::{CloudError, Entitlement, Session, SupabaseClient, UsageSummary};

/// Supabase project URL of Scriba Pro.
pub const SUPABASE_URL: &str = "https://elnrpaedloeequmbdkgy.supabase.co";
/// Supabase publishable (anon) key of Scriba Pro. Public by design.
pub const SUPABASE_ANON_KEY: &str = "sb_publishable_hDRJOngfn37hXEuf6fmbgg__UKeCGFL";

/// The entitlement every beta member holds.
pub const BETA_FEATURE: &str = "beta";
/// The paid entitlement, once it exists.
pub const PRO_FEATURE: &str = "pro";

/// Model proxy of Scriba Pro: speech and assistant calls go through it with
/// the account's session as the key.
pub const PRO_PROXY_URL: &str = "https://scriba-proxy-872870393813.europe-west1.run.app";

/// Seconds of validity left below which the access token is refreshed.
const REFRESH_MARGIN_SECS: i64 = 120;

/// Process-wide account state that model calls need synchronously: which
/// proxy counts as "ours" and the current access token.
#[derive(Debug, Default)]
struct Store {
    proxy_url: Option<String>,
    access_token: Option<String>,
    expires_at: i64,
}

fn store() -> &'static Mutex<Store> {
    static STORE: OnceLock<Mutex<Store>> = OnceLock::new();
    STORE.get_or_init(|| Mutex::new(Store::default()))
}

/// Record the proxy the client should trust. Called on startup and whenever
/// the config changes; cheap, idempotent.
pub fn init(config: &ScribaConfig) {
    let url = proxy_url(config);
    if let Ok(mut s) = store().lock() {
        s.proxy_url = url;
    }
}

fn remember_session(session: &Session) {
    if let Ok(mut s) = store().lock() {
        s.access_token = Some(session.access_token.clone());
        s.expires_at = session.expires_at;
    }
}

fn forget_session() {
    if let Ok(mut s) = store().lock() {
        s.access_token = None;
        s.expires_at = 0;
    }
}

/// The access token to present to the proxy, if the session is live.
pub fn access_token() -> Option<String> {
    let s = store().lock().ok()?;
    let now = chrono::Utc::now().timestamp();
    (s.expires_at > now)
        .then(|| s.access_token.clone())
        .flatten()
}

/// Whether the token will still be valid in a couple of minutes.
fn token_is_fresh() -> bool {
    store()
        .lock()
        .map(|s| {
            s.access_token.is_some()
                && s.expires_at - chrono::Utc::now().timestamp() > REFRESH_MARGIN_SECS
        })
        .unwrap_or(false)
}

/// Base URL of the Scriba Pro model proxy: environment, then config
/// override, then the built-in constant. `None` when not configured or not
/// served over TLS.
pub fn proxy_url(config: &ScribaConfig) -> Option<String> {
    let url = std::env::var("SCRIBA_PRO_PROXY_URL")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| config.cloud.proxy_url.clone())
        .unwrap_or_else(|| PRO_PROXY_URL.to_string());
    let url = url.trim().trim_end_matches('/');
    (!url.is_empty() && is_safe_endpoint(url)).then(|| url.to_string())
}

/// The proxy URL the process trusts right now (set by [`init`]), else what
/// the environment or the binary say.
fn known_proxy_url() -> Option<String> {
    if let Ok(s) = store().lock()
        && let Some(url) = &s.proxy_url
    {
        return Some(url.clone());
    }
    let url = std::env::var("SCRIBA_PRO_PROXY_URL")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| PRO_PROXY_URL.to_string());
    let url = url.trim().trim_end_matches('/');
    (!url.is_empty()).then(|| url.to_string())
}

/// Whether `url` points at the Scriba Pro proxy, meaning the session token
/// is the credential to send.
pub fn is_pro_url(url: &str) -> bool {
    let Some(proxy) = known_proxy_url() else {
        return false;
    };
    let url = url.trim().trim_end_matches('/');
    url == proxy || url.starts_with(&format!("{proxy}/"))
}

/// Speech endpoint behind the proxy (OpenAI-compatible).
pub fn pro_speech_url(config: &ScribaConfig) -> Option<String> {
    proxy_url(config).map(|p| format!("{p}/openai/v1"))
}

/// Assistant endpoint behind the proxy (Anthropic Messages API).
pub fn pro_assistant_url(config: &ScribaConfig) -> Option<String> {
    proxy_url(config).map(|p| format!("{p}/anthropic/v1"))
}

/// Whether the config sends speech or assistant calls through the proxy.
pub fn uses_pro(config: &ScribaConfig) -> bool {
    init(config);
    is_pro_url(&config.transcription_base_url())
        || config
            .enrichment
            .effective_base_url()
            .map(|u| is_pro_url(&u))
            .unwrap_or(false)
}

/// Whether this account may use the proxy: signed in, entitled, and the
/// proxy is configured in this build.
pub fn pro_available(config: &ScribaConfig) -> bool {
    config.cloud.is_signed_in()
        && (config.cloud.has(BETA_FEATURE) || config.cloud.has(PRO_FEATURE))
        && proxy_url(config).is_some()
}

/// Make sure a live access token is in the store when the config uses the
/// proxy. Cheap when the token is fresh. Errors are the caller's to report;
/// a missing token then surfaces as a clear message at the model call.
pub async fn ensure_session(config: &ScribaConfig) -> Result<(), CloudError> {
    if !uses_pro(config) || token_is_fresh() {
        return Ok(());
    }
    let Some(client) = client_for(config) else {
        return Err(CloudError::Other(
            "Scriba Pro is not configured in this build".into(),
        ));
    };
    match resume_session(&client).await {
        Ok(_) => Ok(()),
        Err(CloudError::SessionExpired) => {
            let _ = secrets::delete(&secrets::session_key(client.url()));
            forget_session();
            Err(CloudError::SessionExpired)
        }
        Err(e) => Err(e),
    }
}

/// What to tell the user when a Pro call has no token.
pub const SESSION_HINT: &str =
    "Scriba Pro session expired. Open Settings \u{2192} Scriba Pro \u{2192} Sign in.";

/// Resolve the project the client should talk to: environment, then config
/// overrides, then the built-in constants. `None` when nothing is configured.
pub fn client_for(config: &ScribaConfig) -> Option<SupabaseClient> {
    let url = std::env::var("SCRIBA_SUPABASE_URL")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| config.cloud.supabase_url.clone())
        .unwrap_or_else(|| SUPABASE_URL.to_string());
    let key = std::env::var("SCRIBA_SUPABASE_ANON_KEY")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| config.cloud.supabase_anon_key.clone())
        .unwrap_or_else(|| SUPABASE_ANON_KEY.to_string());
    let url = url.trim();
    if url.is_empty() || key.trim().is_empty() || !is_safe_endpoint(url) {
        return None;
    }
    Some(SupabaseClient::new(url, key.trim()))
}

/// Tokens only travel over TLS, except to a local Supabase for development.
fn is_safe_endpoint(url: &str) -> bool {
    if url.starts_with("https://") {
        return true;
    }
    url.strip_prefix("http://")
        .map(|rest| {
            let host = rest.split(['/', ':']).next().unwrap_or("");
            host == "localhost" || host == "127.0.0.1" || host == "[::1]"
        })
        .unwrap_or(false)
}

/// Whether Scriba Pro is reachable at all from this build.
pub fn is_configured(config: &ScribaConfig) -> bool {
    client_for(config).is_some()
}

/// Version string reported with beta requests.
pub fn client_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Normalize an email the way the server stores it.
pub fn normalize_email(raw: &str) -> String {
    raw.trim().to_lowercase()
}

/// Cheap sanity check before bothering the server.
pub fn looks_like_email(s: &str) -> bool {
    let s = s.trim();
    let Some((local, domain)) = s.split_once('@') else {
        return false;
    };
    !local.is_empty()
        && domain.contains('.')
        && !domain.starts_with('.')
        && !domain.ends_with('.')
        && s.len() <= 254
        && !s.contains(char::is_whitespace)
}

/// What the account screen shows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccountStatus {
    /// This build has no Scriba Pro project configured.
    Unavailable,
    /// Nobody is signed in.
    SignedOut {
        /// When the user asked to join the beta, if they did.
        requested_at: Option<String>,
    },
    /// An account is signed in (the session may still turn out to be stale).
    SignedIn {
        email: String,
        entitlements: Vec<String>,
    },
}

pub fn status(config: &ScribaConfig) -> AccountStatus {
    if !is_configured(config) {
        return AccountStatus::Unavailable;
    }
    match &config.cloud.email {
        Some(email) => AccountStatus::SignedIn {
            email: email.clone(),
            entitlements: config.cloud.entitlements.clone(),
        },
        None => AccountStatus::SignedOut {
            requested_at: config.cloud.beta_requested_at.clone(),
        },
    }
}

/// Outcome of a completed account operation, to be applied to the config by
/// the caller (the TUI owns the config and saves it).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccountEvent {
    /// The beta request was recorded server-side.
    BetaRequested { email: String },
    /// A code was emailed; the user should type it next.
    CodeSent { email: String },
    /// Signed in: session stored, entitlements fetched.
    SignedIn {
        email: String,
        user_id: String,
        entitlements: Vec<String>,
    },
    /// Entitlements refreshed for the signed-in account, with this month's
    /// usage when the server could report it.
    Refreshed {
        entitlements: Vec<String>,
        usage: Option<UsageSummary>,
    },
    /// The stored session no longer works; the account was forgotten locally.
    SessionLost,
    /// Signed out and forgotten locally.
    SignedOut,
}

/// Apply an event to the config. Does not save.
pub fn apply_event(config: &mut ScribaConfig, event: &AccountEvent) {
    let now = chrono::Utc::now().to_rfc3339();
    match event {
        AccountEvent::BetaRequested { .. } => {
            config.cloud.beta_requested_at = Some(now);
        }
        AccountEvent::CodeSent { .. } => {}
        AccountEvent::SignedIn {
            email,
            user_id,
            entitlements,
        } => {
            config.cloud.email = Some(email.clone());
            config.cloud.user_id = Some(user_id.clone());
            config.cloud.entitlements = entitlements.clone();
            config.cloud.entitlements_checked_at = Some(now);
        }
        AccountEvent::Refreshed { entitlements, .. } => {
            config.cloud.entitlements = entitlements.clone();
            config.cloud.entitlements_checked_at = Some(now);
        }
        AccountEvent::SessionLost | AccountEvent::SignedOut => {
            config.cloud.clear_session();
        }
    }
}

fn features(entitlements: Vec<Entitlement>) -> Vec<String> {
    let mut names: Vec<String> = entitlements.into_iter().map(|e| e.feature).collect();
    names.sort();
    names.dedup();
    names
}

/// Ask to join the beta.
pub async fn request_beta(
    client: &SupabaseClient,
    email: &str,
    note: Option<&str>,
) -> Result<AccountEvent, CloudError> {
    let email = normalize_email(email);
    client.request_beta(&email, note, client_version()).await?;
    Ok(AccountEvent::BetaRequested { email })
}

/// Start signing in: email a code.
pub async fn send_code(client: &SupabaseClient, email: &str) -> Result<AccountEvent, CloudError> {
    let email = normalize_email(email);
    client.send_code(&email).await?;
    Ok(AccountEvent::CodeSent { email })
}

/// Finish signing in: verify the code, store the session, fetch entitlements.
pub async fn verify_code(
    client: &SupabaseClient,
    email: &str,
    code: &str,
) -> Result<AccountEvent, CloudError> {
    let email = normalize_email(email);
    let session = client.verify_code(&email, code.trim()).await?;
    secrets::store(&secrets::session_key(client.url()), &session.refresh_token)
        .map_err(|e| CloudError::Other(format!("could not store the session: {e:#}")))?;
    remember_session(&session);
    let entitlements = client
        .entitlements(&session.access_token)
        .await
        .unwrap_or_default();
    Ok(AccountEvent::SignedIn {
        email: if session.email.is_empty() {
            email
        } else {
            session.email
        },
        user_id: session.user_id,
        entitlements: features(entitlements),
    })
}

/// Get a live session from the stored refresh token, rotating it.
pub async fn resume_session(client: &SupabaseClient) -> Result<Session, CloudError> {
    let key = secrets::session_key(client.url());
    let token = secrets::load(&key)
        .map_err(|e| CloudError::Other(format!("could not read the session: {e:#}")))?
        .ok_or(CloudError::SessionExpired)?;
    let session = client.refresh(&token).await?;
    secrets::store(&key, &session.refresh_token)
        .map_err(|e| CloudError::Other(format!("could not store the session: {e:#}")))?;
    remember_session(&session);
    Ok(session)
}

/// Refresh entitlements for the signed-in account. A dead session forgets
/// the local token and reports [`AccountEvent::SessionLost`] rather than an
/// error, so the UI can offer to sign in again.
pub async fn refresh_entitlements(client: &SupabaseClient) -> Result<AccountEvent, CloudError> {
    match resume_session(client).await {
        Ok(session) => {
            let entitlements = client.entitlements(&session.access_token).await?;
            let usage = client
                .usage_this_month(&session.access_token, &session.user_id)
                .await
                .ok();
            Ok(AccountEvent::Refreshed {
                entitlements: features(entitlements),
                usage,
            })
        }
        Err(CloudError::SessionExpired) => {
            let _ = secrets::delete(&secrets::session_key(client.url()));
            forget_session();
            Ok(AccountEvent::SessionLost)
        }
        Err(e) => Err(e),
    }
}

/// Sign out: revoke server-side when possible, always forget locally.
pub async fn sign_out(client: &SupabaseClient) -> Result<AccountEvent, CloudError> {
    if let Ok(session) = resume_session(client).await {
        let _ = client.sign_out(&session.access_token).await;
    }
    secrets::delete(&secrets::session_key(client.url()))
        .map_err(|e| CloudError::Other(format!("could not forget the session: {e:#}")))?;
    forget_session();
    Ok(AccountEvent::SignedOut)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unconfigured_builds_report_unavailable() {
        let config = ScribaConfig::default();
        // The built-in constants are empty in this build; only a config
        // override makes the project resolvable.
        if SUPABASE_URL.is_empty() && std::env::var("SCRIBA_SUPABASE_URL").is_err() {
            assert_eq!(status(&config), AccountStatus::Unavailable);
        }
        let mut config = ScribaConfig::default();
        config.cloud.supabase_url = Some("https://x.supabase.co/".into());
        config.cloud.supabase_anon_key = Some("anon".into());
        let client = client_for(&config).expect("configured");
        assert_eq!(client.url(), "https://x.supabase.co");
        assert_eq!(
            status(&config),
            AccountStatus::SignedOut { requested_at: None }
        );
    }

    #[test]
    fn events_update_the_config() {
        let mut config = ScribaConfig::default();
        apply_event(
            &mut config,
            &AccountEvent::BetaRequested {
                email: "a@b.co".into(),
            },
        );
        assert!(config.cloud.beta_requested_at.is_some());
        apply_event(
            &mut config,
            &AccountEvent::SignedIn {
                email: "a@b.co".into(),
                user_id: "u".into(),
                entitlements: vec!["beta".into()],
            },
        );
        assert!(config.cloud.is_signed_in());
        assert!(config.cloud.has(BETA_FEATURE));
        apply_event(
            &mut config,
            &AccountEvent::Refreshed {
                entitlements: vec![],
                usage: None,
            },
        );
        assert!(!config.cloud.has(BETA_FEATURE));
        apply_event(&mut config, &AccountEvent::SessionLost);
        assert!(!config.cloud.is_signed_in());
        assert!(
            config.cloud.beta_requested_at.is_some(),
            "the request survives sign-out"
        );
    }

    #[test]
    fn overrides_must_use_tls_unless_local() {
        assert!(is_safe_endpoint("https://x.supabase.co"));
        assert!(is_safe_endpoint("http://localhost:54321"));
        assert!(is_safe_endpoint("http://127.0.0.1:54321/"));
        assert!(!is_safe_endpoint("http://x.supabase.co"));
        assert!(!is_safe_endpoint("http://localhost.evil.com"));
        assert!(!is_safe_endpoint("ftp://x"));
        let mut config = ScribaConfig::default();
        config.cloud.supabase_url = Some("http://x.supabase.co".into());
        config.cloud.supabase_anon_key = Some("anon".into());
        assert!(
            client_for(&config).is_none(),
            "plain http override is ignored"
        );
    }

    #[test]
    fn pro_urls_are_recognized_from_the_configured_proxy() {
        let mut config = ScribaConfig::default();
        config.cloud.proxy_url = Some("https://proxy.example.com/".into());
        init(&config);
        assert_eq!(
            pro_speech_url(&config).as_deref(),
            Some("https://proxy.example.com/openai/v1")
        );
        assert!(is_pro_url("https://proxy.example.com/openai/v1"));
        assert!(is_pro_url("https://proxy.example.com/anthropic/v1/"));
        assert!(!is_pro_url("https://proxy.example.com.evil.com/openai/v1"));
        assert!(!is_pro_url("https://api.openai.com/v1"));
        assert!(!pro_available(&config), "signed out");
        config.cloud.email = Some("a@b.co".into());
        config.cloud.entitlements = vec!["beta".into()];
        assert!(pro_available(&config));
        assert!(access_token().is_none(), "no session in tests");
        // Reset so other tests do not see this proxy.
        config.cloud.proxy_url = None;
        init(&config);
    }

    #[test]
    fn email_checks() {
        assert!(looks_like_email("Giovanni@Exein.io "));
        assert_eq!(normalize_email(" Giovanni@Exein.io "), "giovanni@exein.io");
        for bad in ["", "nope", "@x.io", "a@b", "a b@c.io", "a@.io"] {
            assert!(!looks_like_email(bad), "{bad}");
        }
    }

    #[test]
    fn features_are_sorted_and_unique() {
        let list = vec![
            Entitlement {
                feature: "pro".into(),
                expires_at: None,
            },
            Entitlement {
                feature: "beta".into(),
                expires_at: None,
            },
            Entitlement {
                feature: "beta".into(),
                expires_at: None,
            },
        ];
        assert_eq!(features(list), vec!["beta".to_string(), "pro".to_string()]);
    }
}
