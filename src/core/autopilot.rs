//! Meeting autopilot: the long-running orchestration behind automatic meeting
//! detection and recording.
//!
//! Wraps the meeting watcher ([`super::meeting`]) with the full product flow:
//! detect a meeting, optionally ask via the Record/Ignore panel, record with
//! instant stop on mic release, notify, and apply the post-recording cooldown.
//!
//! Runs in two hosts:
//! - `scriba watch` (CLI): foreground, prints status, Ctrl+C maps to the
//!   shutdown channel.
//! - the TUI dashboard: spawned in the background with `quiet: true` (stdout
//!   would corrupt the ratatui screen; problems surface as desktop
//!   notifications instead). See [`spawn_autopilot`].
//!
//! A shared [`RecordingGuard`] prevents double capture: the autopilot skips
//! auto-recording while a manual recording runs, and the TUI blocks manual
//! recording while an auto-recording runs.

use anyhow::Result;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::{Notify, mpsc, watch};

use super::audio::CompressionSettings;
use super::config::ScribaConfig;
use super::meeting::{
    MeetingEvent, MeetingWatcherConfig, capturing_processes, friendly_process_name,
    meeting_signal, notify_event, run_meeting_watcher, watcher_excludes_self,
};
use super::notify;
use super::workflow::WorkflowManager;

/// What kind of recording is in progress.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordingKind {
    Manual,
    Meeting,
}

/// Where the recording is in its lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordingPhase {
    /// Audio is being captured.
    Recording,
    /// Capture stopped; encoding/transcription/enrichment in progress.
    Processing,
}

/// Snapshot of the recording in progress.
#[derive(Debug, Clone)]
pub struct RecordingInfo {
    pub kind: RecordingKind,
    pub phase: RecordingPhase,
    /// App that triggered a meeting recording (e.g. "Arc"), when known.
    pub source: Option<String>,
    pub started: Instant,
    /// When capture stopped (None while still recording). Freezes `elapsed`.
    pub stopped: Option<Instant>,
    /// Recording directory once the captured audio has been saved (capture is
    /// over; finalization is running).
    pub directory: Option<String>,
}

impl RecordingInfo {
    /// Capture duration so far; stops counting once capture ends.
    pub fn elapsed(&self) -> Duration {
        self.stopped
            .unwrap_or_else(Instant::now)
            .saturating_duration_since(self.started)
    }
}

/// Shared "what is being recorded right now" between the autopilot and the
/// TUI: a mutual-exclusion guard (never two captures at once) plus the state
/// the TUI needs for a live, non-blocking recording indicator.
#[derive(Default)]
pub struct RecordingStatus {
    current: Mutex<Option<RecordingInfo>>,
    /// Latest mic level (0.0–1.0), stored as f32 bits.
    level: AtomicU32,
    stop: Notify,
}

impl RecordingStatus {
    /// Claim the recording slot. Returns false if a recording is already in
    /// progress.
    pub fn try_begin(&self, kind: RecordingKind, source: Option<String>) -> bool {
        let mut current = self.current.lock().unwrap();
        if current.is_some() {
            return false;
        }
        *current = Some(RecordingInfo {
            kind,
            phase: RecordingPhase::Recording,
            source,
            started: Instant::now(),
            stopped: None,
            directory: None,
        });
        self.set_level(0.0);
        true
    }

    /// The captured audio was saved to `directory`: capture is over and
    /// finalization (encode/transcribe/enrich) is in progress.
    pub fn set_saved(&self, directory: String) {
        if let Some(info) = self.current.lock().unwrap().as_mut() {
            info.directory = Some(directory);
            info.phase = RecordingPhase::Processing;
            if info.stopped.is_none() {
                info.stopped = Some(Instant::now());
            }
        }
    }

    pub fn set_phase(&self, phase: RecordingPhase) {
        if let Some(info) = self.current.lock().unwrap().as_mut() {
            info.phase = phase;
            if phase == RecordingPhase::Processing && info.stopped.is_none() {
                info.stopped = Some(Instant::now());
            }
        }
    }

    /// Release the slot.
    pub fn end(&self) {
        *self.current.lock().unwrap() = None;
        self.set_level(0.0);
    }

    pub fn snapshot(&self) -> Option<RecordingInfo> {
        self.current.lock().unwrap().clone()
    }

    pub fn is_active(&self) -> bool {
        self.current.lock().unwrap().is_some()
    }

    pub fn set_level(&self, level: f32) {
        self.level.store(level.to_bits(), Ordering::Relaxed);
    }

    pub fn level(&self) -> f32 {
        f32::from_bits(self.level.load(Ordering::Relaxed))
    }

    /// Ask whoever owns the current recording to stop it (the TUI's Ctrl+R
    /// while a meeting is being auto-recorded). Only wakes a live waiter, so
    /// a request with nothing recording is a no-op rather than a stale token.
    pub fn request_stop(&self) {
        self.stop.notify_waiters();
    }

    /// Resolves when [`RecordingStatus::request_stop`] is called while
    /// awaiting.
    pub async fn stop_requested(&self) {
        self.stop.notified().await;
    }
}

/// Shared handle to the [`RecordingStatus`].
pub type RecordingGuard = Arc<RecordingStatus>;

/// Runtime options for the autopilot (config supplies the detection settings).
#[derive(Debug, Clone)]
pub struct AutopilotOptions {
    /// Transcribe recordings after capture.
    pub transcribe: bool,
    /// Handle a single meeting and return (CLI `--once`).
    pub once: bool,
    /// Extra stdout detail (ignored when `quiet`).
    pub verbose: bool,
    /// Suppress all terminal output; report problems via desktop
    /// notifications. Required when hosted inside the TUI.
    pub quiet: bool,
}

impl Default for AutopilotOptions {
    fn default() -> Self {
        Self {
            transcribe: true,
            once: false,
            verbose: false,
            quiet: false,
        }
    }
}

/// Handle to a background autopilot task.
pub struct AutopilotHandle {
    shutdown: watch::Sender<bool>,
    task: tokio::task::JoinHandle<Result<()>>,
}

impl AutopilotHandle {
    /// Ask the autopilot to stop. An in-progress auto-recording is stopped and
    /// finalized; awaiting that is [`AutopilotHandle::join`]'s job.
    pub fn stop(&self) {
        let _ = self.shutdown.send(true);
    }

    pub fn is_finished(&self) -> bool {
        self.task.is_finished()
    }

    /// Stop and wait for the autopilot to wind down (including finalizing an
    /// in-progress recording).
    pub async fn join(self) -> Result<()> {
        self.stop();
        match self.task.await {
            Ok(result) => result,
            Err(e) => Err(anyhow::anyhow!("autopilot task panicked: {e}")),
        }
    }
}

/// Spawn the autopilot as a background task.
pub fn spawn_autopilot(
    config: ScribaConfig,
    opts: AutopilotOptions,
    recording_guard: RecordingGuard,
) -> AutopilotHandle {
    let (shutdown, shutdown_rx) = watch::channel(false);
    let task = tokio::spawn(run_autopilot(config, opts, shutdown_rx, recording_guard));
    AutopilotHandle { shutdown, task }
}

/// Resolves when shutdown is requested (or the sender is dropped, which we
/// treat the same). Safe to re-await after it has fired.
async fn wait_shutdown(rx: &mut watch::Receiver<bool>) {
    loop {
        if *rx.borrow() {
            return;
        }
        if rx.changed().await.is_err() {
            return;
        }
    }
}

/// Surface a problem: stderr in CLI mode, desktop notification when quiet
/// (the TUI owns the terminal).
fn report_issue(quiet: bool, msg: &str) {
    if quiet {
        notify::notify("Scriba \u{00B7} Meeting watch", msg);
    } else {
        eprintln!("⚠️  {msg}");
    }
}

/// What to do with a detected meeting after the optional confirmation step.
enum MeetingDecision {
    Record,
    Skip,
    AlreadyEnded,
}

/// Handle to a running meeting-watcher thread.
struct WatcherHandle {
    stop: Arc<AtomicBool>,
    events: mpsc::Receiver<MeetingEvent>,
    task: tokio::task::JoinHandle<Result<()>>,
}

fn spawn_watcher(config: MeetingWatcherConfig) -> WatcherHandle {
    let (event_tx, events) = mpsc::channel::<MeetingEvent>(8);
    let stop = Arc::new(AtomicBool::new(false));
    let task = tokio::task::spawn_blocking({
        let stop = stop.clone();
        move || run_meeting_watcher(config, event_tx, stop)
    });
    WatcherHandle { stop, events, task }
}

/// The watcher's event channel closed: join the thread and surface why.
async fn watcher_exit_error(watcher: &mut WatcherHandle) -> anyhow::Error {
    match (&mut watcher.task).await {
        Ok(Ok(())) => anyhow::anyhow!("meeting watcher exited unexpectedly"),
        Ok(Err(e)) => e,
        Err(e) => anyhow::anyhow!("meeting watcher task panicked: {e}"),
    }
}

/// Run the meeting autopilot until `shutdown` fires (or one meeting is
/// handled, with `opts.once`).
///
/// Where the OS attributes mic use per process (macOS 14+, PulseAudio/
/// PipeWire), the watcher keeps running during the recording with Scriba's own
/// capture excluded, and the recording is stopped the moment the meeting app
/// releases the mic; the silence timeout is only a fallback net. On older
/// macOS the watcher can't tell Scriba's capture from the meeting's, so it is
/// paused while recording and the silence timeout is the stop mechanism.
///
/// After each recording a cooldown suppresses new detections, so another
/// recording tool reacting to Scriba's capture can't trigger a feedback loop
/// of tiny recordings (`meeting_detection.cooldown_seconds`; listing the tool
/// in `meeting_detection.ignored_processes` removes it from detection
/// entirely).
pub async fn run_autopilot(
    config: ScribaConfig,
    opts: AutopilotOptions,
    mut shutdown: watch::Receiver<bool>,
    recording_guard: RecordingGuard,
) -> Result<()> {
    let quiet = opts.quiet;
    let verbose = opts.verbose && !quiet;
    macro_rules! say {
        ($($arg:tt)*) => {
            if !quiet {
                println!($($arg)*);
            }
        };
    }

    let md = config.meeting_detection.clone();
    if !md.enabled {
        return Ok(());
    }
    let auto_record = md.auto_record;
    let confirm = md.confirm_before_record;
    let silence_fallback = Duration::from_secs(md.min_silence_seconds as u64);
    let cooldown = Duration::from_secs(md.cooldown_seconds as u64);
    let exclude_self = watcher_excludes_self();

    // Build the native notification panel up front so the first detection
    // doesn't wait on swiftc (first run only, ~10s).
    if !notify::notification_helper_ready() {
        say!("   Preparing notification panel (first run, takes a few seconds)...");
    }
    match tokio::task::spawn_blocking(notify::prepare_notification_helper).await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            if !quiet {
                eprintln!(
                    "⚠️  Native notification panel unavailable ({e}); using AppleScript dialogs instead."
                );
            }
        }
        Err(e) => report_issue(quiet, &format!("Notification panel setup task failed: {e}")),
    }

    let watcher_cfg = MeetingWatcherConfig {
        input_device: md.input_device.clone(),
        ignored_processes: md.ignored_processes.clone(),
        verbose,
    };

    // Surface anything already holding the mic: a meeting in progress won't be
    // detected until it ends, and another recording tool showing up here is a
    // candidate for `meeting_detection.ignored_processes`.
    if !quiet
        && let Ok(procs) = capturing_processes(&watcher_cfg)
        && !procs.is_empty()
    {
        say!(
            "   ⚠️  Mic currently in use by: {} — a meeting already in progress is not detected until it ends.",
            procs.join(", ")
        );
    }

    let mut watcher = spawn_watcher(watcher_cfg.clone());

    // Set when an auto-recording stopped (silence fallback) before the meeting
    // app released the mic: the next MeetingEnded should still notify.
    let mut pending_end_notify = false;
    // After a recording finishes, suppress new detections until this instant
    // (breaks feedback loops with other recording tools reacting to us).
    let mut cooldown_until: Option<tokio::time::Instant> = None;

    'outer: loop {
        // Phase 1: wait for a meeting to start.
        loop {
            tokio::select! {
                biased;
                _ = wait_shutdown(&mut shutdown) => break 'outer,
                evt = watcher.events.recv() => match evt {
                    Some(MeetingEvent::MeetingStarted) => {
                        match cooldown_until {
                            Some(until) if tokio::time::Instant::now() < until => {
                                if verbose {
                                    say!("🧊 Detection during post-recording cooldown — waiting it out.");
                                }
                                tokio::select! {
                                    biased;
                                    _ = wait_shutdown(&mut shutdown) => break 'outer,
                                    _ = tokio::time::sleep_until(until) => {}
                                }
                                cooldown_until = None;
                                // Discard events raced during the cooldown and
                                // judge by the current state instead.
                                while watcher.events.try_recv().is_ok() {}
                                if meeting_signal(&watcher_cfg).unwrap_or(false) {
                                    break; // outlasted the cooldown: a real meeting
                                }
                                // Fizzled during cooldown: keep waiting.
                            }
                            _ => {
                                cooldown_until = None;
                                break;
                            }
                        }
                    }
                    Some(MeetingEvent::MeetingEnded) => {
                        if pending_end_notify {
                            pending_end_notify = false;
                            notify_event(MeetingEvent::MeetingEnded, true, None);
                            if opts.once {
                                break 'outer;
                            }
                        }
                    }
                    None => {
                        let err = watcher_exit_error(&mut watcher).await;
                        report_issue(quiet, &format!("Meeting detection stopped: {err}"));
                        return Err(err);
                    }
                },
            }
        }

        // Which app triggered the detection, as a display name ("Arc", not
        // "company.thebrowser.browser.helper") for the panel, notifications
        // and the TUI indicator.
        let trigger = capturing_processes(&watcher_cfg)
            .ok()
            .filter(|p| !p.is_empty())
            .map(|p| {
                let mut names: Vec<String> = p.iter().map(|n| friendly_process_name(n)).collect();
                names.dedup();
                names.join(", ")
            });

        // Notification-only mode: announce start and end, never record.
        if !auto_record {
            notify_event(MeetingEvent::MeetingStarted, false, trigger.as_deref());
            if !wait_for_meeting_end(&mut watcher, &mut shutdown, quiet).await? {
                break;
            }
            if opts.once {
                break;
            }
            continue;
        }

        let decision = if confirm {
            // The panel shows this as the subtitle and resolves bundle IDs to
            // app names (e.g. "company.thebrowser.browser.helper" -> "Arc").
            let subtitle = trigger.clone().unwrap_or_default();
            let dialog = notify::confirm(
                "Scriba \u{00B7} Meeting detected",
                &subtitle,
                "Record",
                "Ignore",
                md.confirm_timeout_seconds,
                false,
            );
            tokio::pin!(dialog);
            loop {
                tokio::select! {
                    biased;
                    _ = wait_shutdown(&mut shutdown) => break 'outer,
                    answer = &mut dialog => {
                        break if answer { MeetingDecision::Record } else { MeetingDecision::Skip };
                    }
                    evt = watcher.events.recv() => match evt {
                        // Meeting over before the user answered: the dropped
                        // dialog future kills the dialog process.
                        Some(MeetingEvent::MeetingEnded) => break MeetingDecision::AlreadyEnded,
                        Some(_) => {}
                        None => {
                            let err = watcher_exit_error(&mut watcher).await;
                            report_issue(quiet, &format!("Meeting detection stopped: {err}"));
                            return Err(err);
                        }
                    },
                }
            }
        } else {
            notify_event(MeetingEvent::MeetingStarted, true, trigger.as_deref());
            MeetingDecision::Record
        };

        // A manual recording is already running: never double-capture.
        let decision = match decision {
            MeetingDecision::Record
                if !recording_guard.try_begin(RecordingKind::Meeting, trigger.clone()) =>
            {
                if verbose {
                    say!(
                        "🎙️  A manual recording is in progress — not auto-recording this meeting."
                    );
                }
                MeetingDecision::Skip
            }
            other => other,
        };

        match decision {
            MeetingDecision::AlreadyEnded => {
                notify_event(MeetingEvent::MeetingEnded, false, None);
                if opts.once {
                    break;
                }
                continue;
            }
            MeetingDecision::Skip => {
                if verbose {
                    say!("🙈 Not recording this meeting.");
                }
                if !wait_for_meeting_end(&mut watcher, &mut shutdown, quiet).await? {
                    break;
                }
                if opts.once {
                    break;
                }
                continue;
            }
            MeetingDecision::Record => {}
        }
        // recording_guard is held from here until the recording finishes.

        if !exclude_self {
            // Our own capture would read as "mic in use": pause the watcher
            // for the duration of the recording. The thread exits within one
            // poll interval; no need to join it here.
            watcher.stop.store(true, Ordering::Relaxed);
        }

        if verbose {
            if exclude_self {
                say!(
                    "🎙️  Recording meeting (stops on mic release; silence fallback {}s)...",
                    md.min_silence_seconds
                );
            } else {
                say!(
                    "🎙️  Recording meeting (auto-stop after {}s of silence)...",
                    md.min_silence_seconds
                );
            }
        }

        let (stop_tx, stop_rx) = mpsc::channel::<()>(1);
        let transcribe = opts.transcribe;
        let transcription_mode = if transcribe {
            Some(config.transcription.clone())
        } else {
            None
        };
        let workflow = match WorkflowManager::with_config(config.clone()) {
            Ok(w) => w,
            Err(e) => {
                recording_guard.end();
                report_issue(quiet, &format!("Meeting recording could not start: {e}"));
                if !wait_for_meeting_end(&mut watcher, &mut shutdown, quiet).await? {
                    break;
                }
                continue;
            }
        };
        let mut workflow = workflow;
        let silence_net = Some(silence_fallback);
        // The recording must run on its own task: `record_audio` blocks its
        // task for the whole recording, so polling it inline would starve the
        // watcher-event and shutdown select arms.
        // Feed live mic levels to the shared status for the TUI indicator.
        let (level_tx, mut level_rx) = mpsc::channel::<f32>(64);
        let level_sink = recording_guard.clone();
        tokio::spawn(async move {
            while let Some(level) = level_rx.recv().await {
                level_sink.set_level(level);
            }
        });
        // Fires as soon as the raw recording is saved, so the TUI can show the
        // new recording (and its processing spinner) without waiting for
        // transcription to finish.
        let (saved_tx, mut saved_rx) = tokio::sync::oneshot::channel::<String>();
        let rec_verbose = !quiet;
        let mut rec_task = tokio::spawn(async move {
            workflow
                .record_meeting(
                    Some("Meeting".to_string()),
                    Some(CompressionSettings::speech_optimized()),
                    transcribe,
                    transcription_mode,
                    stop_rx,
                    Some(level_tx),
                    Some(saved_tx),
                    silence_net,
                    rec_verbose,
                )
                .await
        });

        let mut meeting_ended = false;
        let mut interrupted = false;
        let mut user_stopped = false;
        let mut saved_seen = false;
        let mut watcher_alive = exclude_self;
        let rec_result = loop {
            tokio::select! {
                biased;
                saved = &mut saved_rx, if !saved_seen => {
                    saved_seen = true;
                    if let Ok(directory) = saved {
                        recording_guard.set_saved(directory);
                    }
                }
                _ = wait_shutdown(&mut shutdown), if !interrupted => {
                    interrupted = true;
                    recording_guard.set_phase(RecordingPhase::Processing);
                    let _ = stop_tx.try_send(());
                    say!("🛑 Stopping meeting recording...");
                }
                res = &mut rec_task => break res,
                _ = recording_guard.stop_requested(), if !user_stopped && !meeting_ended => {
                    // Stopped from the TUI (Ctrl+R); the meeting itself may
                    // go on, so the end notification waits for the mic release.
                    user_stopped = true;
                    recording_guard.set_phase(RecordingPhase::Processing);
                    let _ = stop_tx.try_send(());
                }
                evt = watcher.events.recv(), if watcher_alive && !meeting_ended => match evt {
                    Some(MeetingEvent::MeetingEnded) => {
                        meeting_ended = true;
                        recording_guard.set_phase(RecordingPhase::Processing);
                        // Notify right away — finalization (encode, DB,
                        // transcription) can take a while.
                        notify_event(MeetingEvent::MeetingEnded, true, None);
                        if verbose {
                            say!("📴 Meeting app released the mic — stopping recording.");
                        }
                        let _ = stop_tx.try_send(());
                    }
                    Some(_) => {}
                    None => watcher_alive = false,
                },
            }
        };
        recording_guard.end();

        match rec_result {
            Ok(Ok(_)) => {
                if verbose {
                    say!("✅ Meeting recording finished.");
                }
            }
            Ok(Err(e)) => report_issue(quiet, &format!("Meeting recording failed: {e}")),
            Err(e) => report_issue(quiet, &format!("Meeting recording task panicked: {e}")),
        }

        if interrupted {
            break;
        }

        cooldown_until = Some(tokio::time::Instant::now() + cooldown);

        if exclude_self {
            if meeting_ended {
                // End notification already fired the moment the mic was
                // released.
                if opts.once {
                    break;
                }
            } else {
                // The silence fallback (or an error) ended the recording while
                // the meeting app still holds the mic; notify once it lets go.
                pending_end_notify = true;
            }
        } else {
            // The watcher was paused, so poll for the mic release ourselves,
            // giving our own just-closed stream a moment to disappear.
            tokio::time::sleep(Duration::from_millis(1500)).await;
            loop {
                if !meeting_signal(&watcher_cfg).unwrap_or(false) {
                    break;
                }
                tokio::select! {
                    biased;
                    _ = wait_shutdown(&mut shutdown) => break 'outer,
                    _ = tokio::time::sleep(Duration::from_secs(1)) => {}
                }
            }
            notify_event(MeetingEvent::MeetingEnded, true, None);
            if opts.once {
                break;
            }
            watcher = spawn_watcher(watcher_cfg.clone());
        }
    }

    watcher.stop.store(true, Ordering::Relaxed);
    Ok(())
}

/// Wait for the current meeting to end (notifying, `recorded = false` wording)
/// or for shutdown. Returns `Ok(true)` when the meeting ended, `Ok(false)` on
/// shutdown.
async fn wait_for_meeting_end(
    watcher: &mut WatcherHandle,
    shutdown: &mut watch::Receiver<bool>,
    quiet: bool,
) -> Result<bool> {
    loop {
        tokio::select! {
            biased;
            _ = wait_shutdown(shutdown) => return Ok(false),
            evt = watcher.events.recv() => match evt {
                Some(MeetingEvent::MeetingEnded) => {
                    notify_event(MeetingEvent::MeetingEnded, false, None);
                    return Ok(true);
                }
                Some(_) => {}
                None => {
                    let err = watcher_exit_error(watcher).await;
                    report_issue(quiet, &format!("Meeting detection stopped: {err}"));
                    return Err(err);
                }
            },
        }
    }
}
