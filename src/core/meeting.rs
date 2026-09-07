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
//! Processes listed in `ignored_processes` (case-insensitive substring match
//! on the bundle ID / process name) never count as a meeting. This matters
//! when another recording tool runs on the same machine: each tool's recording
//! looks like a meeting to the other, producing a feedback loop of tiny
//! recordings unless one of them is ignored.
//!
//! Platform implementations:
//! - **macOS 14+**: Core Audio process objects
//!   (`kAudioHardwarePropertyProcessObjectList` +
//!   `kAudioProcessPropertyIsRunningInput`), excluding Scriba's own PID. This
//!   attribution keeps working *while Scriba records*, so a meeting's end is
//!   detected the moment the meeting app releases the mic.
//! - **older macOS**: falls back to `kAudioDevicePropertyDeviceIsRunningSomewhere`
//!   across all input devices. This cannot exclude Scriba's own capture (nor
//!   apply the ignore list), so callers must pause the watcher while recording
//!   (see [`watcher_excludes_self`]).
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
    /// Processes whose mic capture never counts as a meeting
    /// (case-insensitive substring match on bundle ID / process name).
    pub ignored_processes: Vec<String>,
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

/// One-shot query: is the meeting signal currently active (a non-ignored
/// process other than us capturing a mic)?
pub fn meeting_signal(config: &MeetingWatcherConfig) -> Result<bool> {
    Ok(!capturing_processes(config)?.is_empty())
}

/// One-shot query: names of the processes currently driving the meeting
/// signal (after PID/ignore/source filtering). Empty = no meeting.
pub fn capturing_processes(config: &MeetingWatcherConfig) -> Result<Vec<String>> {
    platform::capturing_others(&config.input_device, &config.ignored_processes)
}

/// System processes that capture the mic for reasons that are never a
/// meeting. `com.apple.CoreSpeech` in particular grabs the mic right after
/// any recording ends (Siri/dictation re-arming), which would otherwise
/// read as a new meeting and loop forever.
const BUILTIN_IGNORED: &[&str] = &[
    "com.apple.corespeech",
    "com.apple.siri",
    "com.apple.dictation",
    "com.apple.assistant",
];

/// Case-insensitive substring match of a process name against the built-in
/// and user-configured ignore lists.
fn is_ignored(name: &str, ignored: &[String]) -> bool {
    let lower = name.to_lowercase();
    BUILTIN_IGNORED.iter().any(|p| lower.contains(p))
        || ignored
            .iter()
            .any(|p| !p.is_empty() && lower.contains(&p.to_lowercase()))
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
    let initial = platform::capturing_others(&config.input_device, &config.ignored_processes)?;
    if config.verbose {
        if initial.is_empty() {
            println!("   initial mic-in-use-by-others = false");
        } else {
            println!(
                "   initial mic-in-use-by-others = true ({})",
                initial.join(", ")
            );
        }
    }
    let mut debouncer = Debouncer::new(!initial.is_empty());
    let mut last_names = initial;

    while !stop.load(Ordering::Relaxed) {
        if event_tx.is_closed() {
            return Ok(());
        }
        std::thread::sleep(platform::POLL);
        // Transient probe errors (e.g. a device disappearing mid-read) keep
        // the previous state rather than fabricating a transition.
        let now = match platform::capturing_others(&config.input_device, &config.ignored_processes)
        {
            Ok(names) => {
                last_names = names;
                !last_names.is_empty()
            }
            Err(_) => debouncer.settled,
        };
        if let Some(evt) = debouncer.update(now) {
            if config.verbose {
                match evt {
                    MeetingEvent::MeetingStarted => println!(
                        "mic-in-use transition: MeetingStarted ({})",
                        last_names.join(", ")
                    ),
                    MeetingEvent::MeetingEnded => println!("mic-in-use transition: MeetingEnded"),
                }
            }
            let _ = event_tx.try_send(evt);
        }
    }
    Ok(())
}

/// Fire the desktop notification for a meeting event. `recorded` selects the
/// wording (whether Scriba is/was recording the meeting or only observing);
/// `detail` names the app that triggered the detection.
pub fn notify_event(event: MeetingEvent, recorded: bool, detail: Option<&str>) {
    let suffix = detail.map(|d| format!(" ({d})")).unwrap_or_default();
    let (title, body) = match (event, recorded) {
        (MeetingEvent::MeetingStarted, true) => (
            "Scriba \u{00B7} Meeting detected",
            format!("A meeting seems to have started{suffix}. Scriba is recording it."),
        ),
        (MeetingEvent::MeetingStarted, false) => (
            "Scriba \u{00B7} Meeting detected",
            format!("A meeting seems to have started{suffix}."),
        ),
        (MeetingEvent::MeetingEnded, true) => (
            "Scriba \u{00B7} Meeting ended",
            "The meeting ended. Recording has stopped.".to_string(),
        ),
        (MeetingEvent::MeetingEnded, false) => (
            "Scriba \u{00B7} Meeting ended",
            "The meeting ended.".to_string(),
        ),
    };
    super::notify::notify(title, &body);
}

// ─────────────────────────────────────────────────────────────────────────────
// Platform dispatch
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
mod platform {
    use super::*;
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
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

    // CoreFoundation, for reading CFString-valued properties (bundle IDs).
    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        fn CFStringGetCString(
            theString: *const std::ffi::c_void,
            buffer: *mut std::ffi::c_char,
            bufferSize: isize,
            encoding: u32,
        ) -> u8;
        fn CFRelease(cf: *const std::ffi::c_void);
    }
    const CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;

    // Four-char-code selectors and scopes (packed big-endian).
    const SEL_DEVICES: u32 = u32::from_be_bytes(*b"dev#");
    const SEL_RUNNING_SOMEWHERE: u32 = u32::from_be_bytes(*b"irun");
    const SEL_STREAM_CONFIG: u32 = u32::from_be_bytes(*b"slay");
    // Process objects (macOS 14+): per-process audio activity.
    const SEL_PROCESS_OBJECT_LIST: u32 = u32::from_be_bytes(*b"prs#");
    const SEL_PROCESS_PID: u32 = u32::from_be_bytes(*b"ppid");
    const SEL_PROCESS_BUNDLE_ID: u32 = u32::from_be_bytes(*b"pbid");
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

    /// Names of processes (other than us, not ignored) currently capturing
    /// mic input. On pre-14 macOS the device-level fallback cannot attribute
    /// usage, so an active mic reports a single "unknown process" entry and
    /// the ignore list has no effect. The device filter is ignored on macOS.
    pub fn capturing_others(_filter: &Option<String>, ignored: &[String]) -> Result<Vec<String>> {
        if process_api_available() {
            let own_pid = std::process::id();
            let procs = get_audio_objects(SYSTEM_OBJECT, SEL_PROCESS_OBJECT_LIST, SCOPE_GLOBAL)?;
            let mut names = Vec::new();
            for proc_obj in procs {
                let pid = get_prop_u32(proc_obj, SEL_PROCESS_PID).unwrap_or(0);
                if pid == own_pid {
                    continue;
                }
                if get_prop_u32(proc_obj, SEL_PROCESS_IS_RUNNING_INPUT).unwrap_or(0) == 0 {
                    continue;
                }
                let name = display_name(pid, proc_obj);
                if is_ignored(&name, ignored) {
                    continue;
                }
                names.push(name);
            }
            Ok(names)
        } else {
            let devices = enumerate_input_devices()?;
            if compute_in_use(&devices)? {
                Ok(vec!["unknown process".to_string()])
            } else {
                Ok(Vec::new())
            }
        }
    }

    /// Human-readable name for a capturing process: bundle ID when the HAL
    /// knows it, executable name otherwise. Cached per PID (PIDs recycle
    /// rarely and the probe runs several times a second).
    fn display_name(pid: u32, proc_obj: AudioObjectID) -> String {
        static CACHE: OnceLock<Mutex<HashMap<u32, String>>> = OnceLock::new();
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(name) = cache.lock().unwrap().get(&pid) {
            return name.clone();
        }
        let name = process_bundle_id(proc_obj)
            .or_else(|| process_name_from_ps(pid))
            .unwrap_or_else(|| format!("pid {pid}"));
        cache.lock().unwrap().insert(pid, name.clone());
        name
    }

    /// Read `kAudioProcessPropertyBundleID` (a CFString) from a process object.
    fn process_bundle_id(proc_obj: AudioObjectID) -> Option<String> {
        let addr = AudioObjectPropertyAddress {
            mSelector: SEL_PROCESS_BUNDLE_ID,
            mScope: SCOPE_GLOBAL,
            mElement: ELEMENT_WILDCARD,
        };
        let mut cf: *const std::ffi::c_void = std::ptr::null();
        let mut size = std::mem::size_of::<*const std::ffi::c_void>() as u32;
        let status = unsafe {
            AudioObjectGetPropertyData(
                proc_obj,
                &addr,
                0,
                std::ptr::null(),
                &mut size,
                &mut cf as *mut *const std::ffi::c_void as *mut std::ffi::c_void,
            )
        };
        if status != 0 || cf.is_null() {
            return None;
        }
        let mut buf = [0u8; 256];
        let ok = unsafe {
            CFStringGetCString(
                cf,
                buf.as_mut_ptr() as *mut std::ffi::c_char,
                buf.len() as isize,
                CF_STRING_ENCODING_UTF8,
            )
        };
        unsafe { CFRelease(cf) };
        if ok == 0 {
            return None;
        }
        let s = std::ffi::CStr::from_bytes_until_nul(&buf)
            .ok()?
            .to_string_lossy()
            .trim()
            .to_string();
        if s.is_empty() { None } else { Some(s) }
    }

    /// Fallback name via `ps` for processes without a bundle ID (CLI tools).
    fn process_name_from_ps(pid: u32) -> Option<String> {
        if pid == 0 {
            return None;
        }
        let out = std::process::Command::new("ps")
            .args(["-p", &pid.to_string(), "-o", "comm="])
            .output()
            .ok()?;
        let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if s.is_empty() {
            None
        } else {
            Some(s.rsplit('/').next().unwrap_or(&s).to_string())
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
            let _ = capturing_others(&None, &[]);
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

    /// Names of processes (other than us, not ignored) capturing from a
    /// non-monitor source. When `name_filter` is set, only count sources
    /// whose name contains it.
    pub fn capturing_others(
        name_filter: &Option<String>,
        ignored: &[String],
    ) -> Result<Vec<String>> {
        let sources = source_names_by_index()?;
        let output = Command::new("pactl")
            .args(["list", "source-outputs"])
            .output()?;
        if !output.status.success() {
            return Err(anyhow::anyhow!("pactl exited non-zero"));
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        let own_pid = std::process::id().to_string();
        Ok(foreign_captures(
            &stdout,
            &sources,
            &own_pid,
            name_filter,
            ignored,
        ))
    }

    /// Pure parse of `pactl list source-outputs` output: application names of
    /// source-outputs that belong to another process, are not corked, capture
    /// from a non-monitor source, match the optional source-name filter, and
    /// are not in the ignore list.
    fn foreign_captures(
        listing: &str,
        sources: &HashMap<String, String>,
        own_pid: &str,
        name_filter: &Option<String>,
        ignored: &[String],
    ) -> Vec<String> {
        let mut names = Vec::new();
        for block in listing.split("Source Output #").skip(1) {
            let mut source_index = None;
            let mut pid = None;
            let mut app_name = None;
            let mut corked = false;
            for line in block.lines() {
                let t = line.trim();
                if let Some(v) = t.strip_prefix("Source: ") {
                    source_index = Some(v.trim().to_string());
                } else if let Some(v) = t.strip_prefix("Corked: ") {
                    corked = v.trim() == "yes";
                } else if let Some(v) = t.strip_prefix("application.process.id = ") {
                    pid = Some(v.trim().trim_matches('"').to_string());
                } else if let Some(v) = t.strip_prefix("application.name = ") {
                    app_name = Some(v.trim().trim_matches('"').to_string());
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
            let name = app_name.unwrap_or_else(|| "unknown process".to_string());
            if is_ignored(&name, ignored) {
                continue;
            }
            names.push(name);
        }
        names
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
        fn detects_foreign_mic_capture_with_name() {
            let names = foreign_captures(LISTING, &sources(), "9999", &None, &[]);
            assert_eq!(names, vec!["ZOOM VoiceEngine".to_string()]);
        }

        #[test]
        fn excludes_own_pid() {
            // Only #57 is a live mic capture; if that's us, nothing remains
            // (#58 is a monitor source, #59 is corked).
            assert!(foreign_captures(LISTING, &sources(), "4242", &None, &[]).is_empty());
        }

        #[test]
        fn ignored_processes_do_not_count() {
            let ignored = vec!["zoom".to_string()];
            assert!(foreign_captures(LISTING, &sources(), "9999", &None, &ignored).is_empty());
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
            assert!(foreign_captures(only_monitor, &sources(), "9999", &None, &[]).is_empty());
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
            assert!(foreign_captures(only_corked, &sources(), "9999", &None, &[]).is_empty());
        }

        #[test]
        fn source_name_filter_applies() {
            assert!(
                !foreign_captures(
                    LISTING,
                    &sources(),
                    "9999",
                    &Some("usb-mic".to_string()),
                    &[]
                )
                .is_empty()
            );
            assert!(
                foreign_captures(
                    LISTING,
                    &sources(),
                    "9999",
                    &Some("builtin".to_string()),
                    &[]
                )
                .is_empty()
            );
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

    pub fn capturing_others(_filter: &Option<String>, _ignored: &[String]) -> Result<Vec<String>> {
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

    #[test]
    fn ignore_matching_is_case_insensitive_substring() {
        let ignored = vec!["granola".to_string(), "us.zoom".to_string()];
        assert!(is_ignored("Granola", &ignored));
        assert!(is_ignored("com.granola.app", &ignored));
        assert!(is_ignored("US.ZOOM.XOS", &ignored));
        assert!(!is_ignored("com.google.Chrome", &ignored));
        assert!(!is_ignored("anything", &[]));
        assert!(!is_ignored("anything", &[String::new()]));
    }

    #[test]
    fn system_speech_daemons_are_always_ignored() {
        assert!(is_ignored("com.apple.CoreSpeech", &[]));
        assert!(is_ignored("com.apple.siri.embeddedspeech", &[]));
        assert!(!is_ignored("us.zoom.xos", &[]));
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
