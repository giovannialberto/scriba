//! Automatic meeting detection for Scriba.
//!
//! Instead of listening to audio levels (which would fire on any casual
//! talking), the watcher detects when a microphone is *in use by another
//! process*. Meeting apps like Zoom and Google Meet open the input device for
//! capture when you join a call and release it when you leave — so "someone
//! else is capturing the mic" is a precise, low-false-positive signal for both
//! the start and the end of a meeting.
//!
//! The watcher polls the platform signal on a short interval and emits events
//! on transitions. Polling (rather than Core Audio property listeners) is
//! deliberate: HAL listener callbacks are delivered on the process's main
//! CFRunLoop, which a CLI never runs, so listeners never fire here. The
//! property reads are cheap and re-enumerating on every poll also picks up
//! hot-plugged devices (e.g. AirPods connected when joining a call).
//!
//! Platform implementations:
//! - **macOS 14+**: Core Audio process objects
//!   (`kAudioHardwarePropertyProcessObjectList` +
//!   `kAudioProcessPropertyIsRunningInput`), excluding Scriba's own PID. This
//!   attribution keeps working *while Scriba records*, so a meeting's end is
//!   detected the moment the meeting app releases the mic.
//! - **older macOS**: falls back to `kAudioDevicePropertyDeviceIsRunningSomewhere`
//!   across all input devices. This cannot exclude Scriba's own capture, so
//!   callers must pause the watcher while recording (see
//!   [`watcher_excludes_self`]).
//! - **Linux**: polls `pactl list source-outputs` (PulseAudio/PipeWire),
//!   excluding Scriba's own PID, corked streams, and monitor sources.
//! - Other platforms: not supported.

use anyhow::Result;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::mpsc;

/// Events emitted by the meeting watcher.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeetingEvent {
    /// A microphone was grabbed by another process — a meeting likely started.
    MeetingStarted,
    /// The other process released the microphone — the meeting likely ended.
    MeetingEnded,
}

/// Configuration snapshot consumed by the watcher.
#[derive(Debug, Clone)]
pub struct MeetingWatcherConfig {
    /// Linux: only count source-outputs on sources whose name contains this
    /// (substring match). `None` = any. Ignored on macOS.
    pub input_device: Option<String>,
    pub verbose: bool,
}

/// Whether the platform signal can distinguish Scriba's own capture from
/// other processes' capture.
///
/// When true, the watcher keeps working while Scriba records, so a meeting's
/// end stops the recording immediately. When false, callers must pause the
/// watcher during recording and rely on the silence fallback to stop.
pub fn watcher_excludes_self() -> bool {
    platform::excludes_self()
}

/// One-shot query: is a microphone currently in use (by another process,
/// where the platform can tell)? Used by the fallback orchestration path to
/// wait for the meeting app to release the mic after a recording finishes.
pub fn mic_in_use_by_others() -> Result<bool> {
    platform::mic_in_use_by_others(&None)
}

/// Map a state transition of the "mic in use by others" signal to an event.
fn transition(prev: bool, now: bool) -> Option<MeetingEvent> {
    match (prev, now) {
        (false, true) => Some(MeetingEvent::MeetingStarted),
        (true, false) => Some(MeetingEvent::MeetingEnded),
        _ => None,
    }
}

/// Polls a changed state must survive before a transition is accepted.
/// Filters sub-second blips (apps briefly probing the mic on startup) without
/// adding noticeable latency (2 polls ≈ 0.5–1s depending on platform).
const CONFIRM_POLLS: u32 = 2;

/// Debounces the raw "mic in use by others" samples into meeting events:
/// a change only becomes a transition after holding for [`CONFIRM_POLLS`]
/// consecutive samples.
struct Debouncer {
    settled: bool,
    streak: u32,
}

impl Debouncer {
    fn new(initial: bool) -> Self {
        Self {
            settled: initial,
            streak: 0,
        }
    }

    fn update(&mut self, now: bool) -> Option<MeetingEvent> {
        if now == self.settled {
            self.streak = 0;
            return None;
        }
        self.streak += 1;
        if self.streak < CONFIRM_POLLS {
            return None;
        }
        let prev = self.settled;
        self.settled = now;
        self.streak = 0;
        transition(prev, now)
    }
}

/// Run the meeting watcher until `stop` is set or `event_tx` is closed.
///
/// Polls the platform's "mic in use by another process" signal and emits
/// `MeetingEvent`s on transitions. Blocking; run it on a dedicated thread
/// (e.g. `spawn_blocking`). Returns an error if the platform signal is
/// unavailable (unsupported OS, no devices, `pactl` missing).
pub fn run_meeting_watcher(
    config: MeetingWatcherConfig,
    event_tx: mpsc::Sender<MeetingEvent>,
    stop: Arc<AtomicBool>,
) -> Result<()> {
    if config.verbose {
        println!(
            "👀 Meeting watcher polling: {}",
            platform::backend_description()
        );
    }
    // The first probe doubles as an availability check: fail fast instead of
    // silently watching a signal that can never change.
    let initial = platform::mic_in_use_by_others(&config.input_device)?;
    if config.verbose {
        println!("   initial mic-in-use-by-others = {initial}");
    }
    let mut debouncer = Debouncer::new(initial);

    while !stop.load(Ordering::Relaxed) {
        if event_tx.is_closed() {
            return Ok(());
        }
        std::thread::sleep(platform::POLL);
        // Transient probe errors (e.g. a device disappearing mid-read) keep
        // the previous state rather than fabricating a transition.
        let now = platform::mic_in_use_by_others(&config.input_device).unwrap_or(debouncer.settled);
        if let Some(evt) = debouncer.update(now) {
            if config.verbose {
                println!("mic-in-use transition: {:?}", evt);
            }
            let _ = event_tx.try_send(evt);
        }
    }
    Ok(())
}

/// Fire the desktop notification for a meeting event. `recorded` selects the
/// wording: whether Scriba is/was recording the meeting or only observing.
pub fn notify_event(event: MeetingEvent, recorded: bool) {
    let (title, body) = match (event, recorded) {
        (MeetingEvent::MeetingStarted, true) => (
            "Scriba \u{00B7} Meeting detected",
            "A meeting seems to have started. Scriba is recording it.",
        ),
        (MeetingEvent::MeetingStarted, false) => (
            "Scriba \u{00B7} Meeting detected",
            "A meeting seems to have started.",
        ),
        (MeetingEvent::MeetingEnded, true) => (
            "Scriba \u{00B7} Meeting ended",
            "The meeting ended. Recording has stopped.",
        ),
        (MeetingEvent::MeetingEnded, false) => {
            ("Scriba \u{00B7} Meeting ended", "The meeting ended.")
        }
    };
    super::notify::notify(title, body);
}

// ─────────────────────────────────────────────────────────────────────────────
// Platform dispatch
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
mod platform {
    use super::*;
    use std::sync::OnceLock;
    use std::time::Duration;

    pub const POLL: Duration = Duration::from_millis(250);

    // Core Audio FFI (CoreAudio.framework). We link against the framework and
    // declare the handful of C functions / constants we need.

    type AudioObjectID = u32;
    type OSStatus = i32;

    #[allow(non_snake_case)]
    #[repr(C)]
    struct AudioObjectPropertyAddress {
        mSelector: u32,
        mScope: u32,
        mElement: u32,
    }

    #[allow(non_snake_case)]
    #[link(name = "CoreAudio", kind = "framework")]
    unsafe extern "C" {
        fn AudioObjectGetPropertyDataSize(
            inObjectID: AudioObjectID,
            inAddress: *const AudioObjectPropertyAddress,
            inQualifierDataSize: u32,
            inQualifierData: *const std::ffi::c_void,
            outDataSize: *mut u32,
        ) -> OSStatus;

        fn AudioObjectGetPropertyData(
            inObjectID: AudioObjectID,
            inAddress: *const AudioObjectPropertyAddress,
            inQualifierDataSize: u32,
            inQualifierData: *const std::ffi::c_void,
            ioDataSize: *mut u32,
            outData: *mut std::ffi::c_void,
        ) -> OSStatus;
    }

    // Four-char-code selectors and scopes (packed big-endian).
    const SEL_DEVICES: u32 = u32::from_be_bytes(*b"dev#");
    const SEL_RUNNING_SOMEWHERE: u32 = u32::from_be_bytes(*b"irun");
    const SEL_STREAM_CONFIG: u32 = u32::from_be_bytes(*b"slay");
    // Process objects (macOS 14+): per-process audio activity.
    const SEL_PROCESS_OBJECT_LIST: u32 = u32::from_be_bytes(*b"prs#");
    const SEL_PROCESS_PID: u32 = u32::from_be_bytes(*b"ppid");
    const SEL_PROCESS_IS_RUNNING_INPUT: u32 = u32::from_be_bytes(*b"piri");

    const SCOPE_GLOBAL: u32 = u32::from_be_bytes(*b"glob");
    const SCOPE_INPUT: u32 = u32::from_be_bytes(*b"inpt");
    const ELEMENT_WILDCARD: u32 = 0;

    const SYSTEM_OBJECT: AudioObjectID = 1;

    /// Whether the per-process audio API (macOS 14+) is available.
    fn process_api_available() -> bool {
        static AVAILABLE: OnceLock<bool> = OnceLock::new();
        *AVAILABLE.get_or_init(|| {
            get_audio_objects(SYSTEM_OBJECT, SEL_PROCESS_OBJECT_LIST, SCOPE_GLOBAL).is_ok()
        })
    }

    pub fn excludes_self() -> bool {
        process_api_available()
    }

    pub fn backend_description() -> &'static str {
        if process_api_available() {
            "Core Audio process objects (per-process mic use, excluding Scriba)"
        } else {
            "Core Audio device state (mic in use by any process)"
        }
    }

    /// Is a microphone in use — by a process other than us where the OS can
    /// tell (macOS 14+), by anyone otherwise. The device filter is ignored on
    /// macOS.
    pub fn mic_in_use_by_others(_filter: &Option<String>) -> Result<bool> {
        if process_api_available() {
            let own_pid = std::process::id();
            let procs = get_audio_objects(SYSTEM_OBJECT, SEL_PROCESS_OBJECT_LIST, SCOPE_GLOBAL)?;
            for proc_obj in procs {
                if get_prop_u32(proc_obj, SEL_PROCESS_PID).ok() == Some(own_pid) {
                    continue;
                }
                if get_prop_u32(proc_obj, SEL_PROCESS_IS_RUNNING_INPUT).unwrap_or(0) != 0 {
                    return Ok(true);
                }
            }
            Ok(false)
        } else {
            let devices = enumerate_input_devices()?;
            compute_in_use(&devices)
        }
    }

    /// Read a 4-byte property value from an audio object (global scope).
    fn get_prop_u32(object: AudioObjectID, selector: u32) -> Result<u32> {
        let addr = AudioObjectPropertyAddress {
            mSelector: selector,
            mScope: SCOPE_GLOBAL,
            mElement: ELEMENT_WILDCARD,
        };
        let mut value: u32 = 0;
        let mut size = std::mem::size_of::<u32>() as u32;
        let status = unsafe {
            AudioObjectGetPropertyData(
                object,
                &addr,
                0,
                std::ptr::null(),
                &mut size,
                &mut value as *mut u32 as *mut std::ffi::c_void,
            )
        };
        if status != 0 {
            return Err(anyhow::anyhow!(
                "AudioObjectGetPropertyData failed (status {status})"
            ));
        }
        Ok(value)
    }

    /// OR of `DeviceIsRunningSomewhere` across the given devices.
    fn compute_in_use(devices: &[AudioObjectID]) -> Result<bool> {
        for dev in devices {
            if get_prop_u32(*dev, SEL_RUNNING_SOMEWHERE).unwrap_or(0) != 0 {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Enumerate all audio devices and keep those that have input streams
    /// (i.e. microphones / input-capable devices).
    fn enumerate_input_devices() -> Result<Vec<AudioObjectID>> {
        let all = get_audio_objects(SYSTEM_OBJECT, SEL_DEVICES, SCOPE_GLOBAL)?;
        let mut inputs = Vec::new();
        for dev in all {
            if has_input_channels(dev)? {
                inputs.push(dev);
            }
        }
        Ok(inputs)
    }

    /// Read the input-scope stream configuration and return true if the device
    /// has any input channels.
    fn has_input_channels(dev: AudioObjectID) -> Result<bool> {
        let addr = AudioObjectPropertyAddress {
            mSelector: SEL_STREAM_CONFIG,
            mScope: SCOPE_INPUT,
            mElement: ELEMENT_WILDCARD,
        };
        let mut size: u32 = 0;
        let status =
            unsafe { AudioObjectGetPropertyDataSize(dev, &addr, 0, std::ptr::null(), &mut size) };
        if status != 0 || size == 0 {
            return Ok(false);
        }
        let mut buf = vec![0u8; size as usize];
        let mut got = size;
        let status = unsafe {
            AudioObjectGetPropertyData(
                dev,
                &addr,
                0,
                std::ptr::null(),
                &mut got,
                buf.as_mut_ptr() as *mut std::ffi::c_void,
            )
        };
        if status != 0 {
            return Ok(false);
        }
        // AudioBufferList layout (64-bit):
        //   mNumberBuffers: u32 at offset 0
        //   4 bytes padding (to align the AudioBuffer pointer)
        //   mBuffers: AudioBuffer[1], each 16 bytes; mNumberChannels at +0
        if buf.len() < 4 {
            return Ok(false);
        }
        let n = u32::from_ne_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
        if n == 0 {
            return Ok(false);
        }
        for i in 0..n {
            let off = 8 + i * 16;
            if off + 4 <= buf.len() {
                let ch = u32::from_ne_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
                if ch > 0 {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    /// Read a property that returns an array of `AudioObjectID` values.
    fn get_audio_objects(
        object: AudioObjectID,
        selector: u32,
        scope: u32,
    ) -> Result<Vec<AudioObjectID>> {
        let addr = AudioObjectPropertyAddress {
            mSelector: selector,
            mScope: scope,
            mElement: ELEMENT_WILDCARD,
        };
        let mut size: u32 = 0;
        let status = unsafe {
            AudioObjectGetPropertyDataSize(object, &addr, 0, std::ptr::null(), &mut size)
        };
        if status != 0 {
            return Err(anyhow::anyhow!(
                "AudioObjectGetPropertyDataSize failed (status {status})"
            ));
        }
        let count = (size as usize) / std::mem::size_of::<AudioObjectID>();
        let mut out = vec![0u32; count];
        let mut got = size;
        let status = unsafe {
            AudioObjectGetPropertyData(
                object,
                &addr,
                0,
                std::ptr::null(),
                &mut got,
                out.as_mut_ptr() as *mut std::ffi::c_void,
            )
        };
        if status != 0 {
            return Err(anyhow::anyhow!(
                "AudioObjectGetPropertyData failed (status {status})"
            ));
        }
        Ok(out)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn mic_probe_does_not_panic() {
            // Best-effort: may fail if no input devices in the test env, but
            // must not panic.
            let _ = mic_in_use_by_others(&None);
        }
    }
}

#[cfg(target_os = "linux")]
mod platform {
    use super::*;
    use std::collections::HashMap;
    use std::process::Command;
    use std::time::Duration;

    pub const POLL: Duration = Duration::from_millis(500);

    pub fn excludes_self() -> bool {
        // pactl reports application.process.id per source-output; if a stream
        // lacks it we conservatively count it as another process.
        true
    }

    pub fn backend_description() -> &'static str {
        "PulseAudio/PipeWire source-outputs (excluding Scriba)"
    }

    /// Is any process other than us capturing from a (non-monitor) source?
    /// When `name_filter` is set, only count sources whose name contains it.
    pub fn mic_in_use_by_others(name_filter: &Option<String>) -> Result<bool> {
        let sources = source_names_by_index()?;
        let output = Command::new("pactl")
            .args(["list", "source-outputs"])
            .output()?;
        if !output.status.success() {
            return Err(anyhow::anyhow!("pactl exited non-zero"));
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        let own_pid = std::process::id().to_string();
        Ok(any_foreign_capture(
            &stdout,
            &sources,
            &own_pid,
            name_filter,
        ))
    }

    /// Pure parse of `pactl list source-outputs` output: true if any
    /// source-output belongs to another process, is not corked, captures from
    /// a non-monitor source, and matches the optional source-name filter.
    fn any_foreign_capture(
        listing: &str,
        sources: &HashMap<String, String>,
        own_pid: &str,
        name_filter: &Option<String>,
    ) -> bool {
        for block in listing.split("Source Output #").skip(1) {
            let mut source_index = None;
            let mut pid = None;
            let mut corked = false;
            for line in block.lines() {
                let t = line.trim();
                if let Some(v) = t.strip_prefix("Source: ") {
                    source_index = Some(v.trim().to_string());
                } else if let Some(v) = t.strip_prefix("Corked: ") {
                    corked = v.trim() == "yes";
                } else if let Some(v) = t.strip_prefix("application.process.id = ") {
                    pid = Some(v.trim().trim_matches('"').to_string());
                }
            }
            if corked {
                continue;
            }
            if pid.as_deref() == Some(own_pid) {
                continue;
            }
            let source_name = source_index
                .as_ref()
                .and_then(|i| sources.get(i))
                .map(String::as_str)
                .unwrap_or("");
            // Monitor sources carry system-audio loopback (screen recorders
            // etc.), not a microphone — never a meeting signal.
            if source_name.ends_with(".monitor") {
                continue;
            }
            if let Some(filter) = name_filter {
                if !source_name.to_lowercase().contains(&filter.to_lowercase()) {
                    continue;
                }
            }
            return true;
        }
        false
    }

    /// Map of source index -> source name from `pactl list sources short`.
    fn source_names_by_index() -> Result<HashMap<String, String>> {
        let output = Command::new("pactl")
            .args(["list", "sources", "short"])
            .output()?;
        if !output.status.success() {
            return Err(anyhow::anyhow!("pactl exited non-zero"));
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut map = HashMap::new();
        for line in stdout.lines() {
            let mut fields = line.split_whitespace();
            if let (Some(idx), Some(name)) = (fields.next(), fields.next()) {
                map.insert(idx.to_string(), name.to_string());
            }
        }
        Ok(map)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        const LISTING: &str = "\
Source Output #57
\tDriver: protocol-native.c
\tSource: 3
\tCorked: no
\tProperties:
\t\tapplication.name = \"ZOOM VoiceEngine\"
\t\tapplication.process.id = \"4242\"
Source Output #58
\tDriver: protocol-native.c
\tSource: 5
\tCorked: no
\tProperties:
\t\tapplication.name = \"OBS\"
\t\tapplication.process.id = \"5151\"
Source Output #59
\tDriver: protocol-native.c
\tSource: 3
\tCorked: yes
\tProperties:
\t\tapplication.name = \"Paused App\"
\t\tapplication.process.id = \"6161\"
";

        fn sources() -> HashMap<String, String> {
            HashMap::from([
                ("3".to_string(), "alsa_input.usb-mic".to_string()),
                ("5".to_string(), "alsa_output.hdmi.monitor".to_string()),
            ])
        }

        #[test]
        fn detects_foreign_mic_capture() {
            assert!(any_foreign_capture(LISTING, &sources(), "9999", &None));
        }

        #[test]
        fn excludes_own_pid() {
            // Only #57 is a live mic capture; if that's us, nothing remains
            // (#58 is a monitor source, #59 is corked).
            assert!(!any_foreign_capture(LISTING, &sources(), "4242", &None));
        }

        #[test]
        fn monitor_sources_do_not_count() {
            let only_monitor = "\
Source Output #58
\tSource: 5
\tCorked: no
\tProperties:
\t\tapplication.process.id = \"5151\"
";
            assert!(!any_foreign_capture(
                only_monitor,
                &sources(),
                "9999",
                &None
            ));
        }

        #[test]
        fn corked_streams_do_not_count() {
            let only_corked = "\
Source Output #59
\tSource: 3
\tCorked: yes
\tProperties:
\t\tapplication.process.id = \"6161\"
";
            assert!(!any_foreign_capture(only_corked, &sources(), "9999", &None));
        }

        #[test]
        fn source_name_filter_applies() {
            assert!(any_foreign_capture(
                LISTING,
                &sources(),
                "9999",
                &Some("usb-mic".to_string())
            ));
            assert!(!any_foreign_capture(
                LISTING,
                &sources(),
                "9999",
                &Some("builtin".to_string())
            ));
        }
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
mod platform {
    use super::*;
    use std::time::Duration;

    pub const POLL: Duration = Duration::from_secs(1);

    pub fn excludes_self() -> bool {
        false
    }

    pub fn backend_description() -> &'static str {
        "unsupported platform"
    }

    pub fn mic_in_use_by_others(_filter: &Option<String>) -> Result<bool> {
        Err(anyhow::anyhow!(
            "Meeting detection is only supported on macOS and Linux"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transition_emits_only_on_change() {
        assert_eq!(transition(false, true), Some(MeetingEvent::MeetingStarted));
        assert_eq!(transition(true, false), Some(MeetingEvent::MeetingEnded));
        assert_eq!(transition(false, false), None);
        assert_eq!(transition(true, true), None);
    }

    #[test]
    fn debouncer_requires_state_to_hold() {
        let mut d = Debouncer::new(false);
        assert_eq!(d.update(true), None); // first changed sample: not yet
        assert_eq!(d.update(true), Some(MeetingEvent::MeetingStarted));
        assert_eq!(d.update(true), None); // steady state: no repeat
        assert_eq!(d.update(false), None);
        assert_eq!(d.update(false), Some(MeetingEvent::MeetingEnded));
    }

    #[test]
    fn debouncer_filters_blips() {
        let mut d = Debouncer::new(false);
        assert_eq!(d.update(true), None); // blip: one sample high...
        assert_eq!(d.update(false), None); // ...back low before confirming
        assert_eq!(d.update(false), None);
        assert_eq!(d.update(true), None);
        assert_eq!(d.update(true), Some(MeetingEvent::MeetingStarted));
    }

    #[tokio::test]
    async fn event_channel_roundtrip() {
        let (tx, mut rx) = mpsc::channel::<MeetingEvent>(4);
        tx.send(MeetingEvent::MeetingStarted).await.unwrap();
        tx.send(MeetingEvent::MeetingEnded).await.unwrap();
        assert_eq!(rx.recv().await, Some(MeetingEvent::MeetingStarted));
        assert_eq!(rx.recv().await, Some(MeetingEvent::MeetingEnded));
    }
}
