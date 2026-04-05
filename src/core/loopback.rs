//! Platform-abstracted system audio loopback capture.
//!
//! On macOS, uses ScreenCaptureKit via objc2 bindings to capture system audio
//! natively (no virtual audio driver or Xcode needed). On Linux, uses cpal with
//! PulseAudio/PipeWire monitor sources.

#[cfg(target_os = "linux")]
use anyhow::Context;
use anyhow::Result;
use std::sync::{Arc, Mutex};

use super::audio::{create_encoder, AudioEncoder, AudioFormat, CompressionSettings};

/// Type alias matching the encoder handle pattern used in recording.rs.
type AudioEncoderHandle = Arc<Mutex<Option<Box<dyn AudioEncoder>>>>;

/// Handle to a running loopback capture session.
/// Drop this to stop capture.
pub struct LoopbackHandle {
    _inner: LoopbackInner,
}

impl LoopbackHandle {
    /// Stop the loopback capture and finalize the encoder.
    pub fn stop_and_finalize(self, encoder: &AudioEncoderHandle) -> Result<()> {
        // Dropping _inner stops the capture stream.
        drop(self._inner);
        // Finalize the encoder.
        if let Some(mut enc) = encoder.lock().unwrap().take() {
            enc.finalize()?;
        }
        Ok(())
    }
}

/// Result from starting loopback capture, including the sample rate
/// used by the loopback device (needed for merge/resample).
pub struct LoopbackSession {
    pub handle: LoopbackHandle,
    pub encoder: AudioEncoderHandle,
    pub sample_rate: u32,
    pub channels: u16,
}

// ── macOS: ScreenCaptureKit via objc2 bindings ──────────────────────────

#[cfg(target_os = "macos")]
mod platform {
    use super::*;
    use block2::RcBlock;
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2::{define_class, msg_send, AllocAnyThread, DeclaredClass};
    use objc2_core_media::CMSampleBuffer;
    use objc2_foundation::{NSArray, NSError, NSObject, NSObjectProtocol};
    use objc2_screen_capture_kit::{
        SCContentFilter, SCDisplay, SCShareableContent, SCStream, SCStreamConfiguration,
        SCStreamOutput, SCStreamOutputType,
    };
    use std::cell::UnsafeCell;

    /// Ivars for our SCStreamOutput implementation.
    struct AudioOutputIvars {
        encoder: AudioEncoderHandle,
    }

    define_class!(
        #[unsafe(super(NSObject))]
        #[name = "ScribaAudioOutput"]
        #[ivars = AudioOutputIvars]
        struct ScribaAudioOutput;

        unsafe impl NSObjectProtocol for ScribaAudioOutput {}

        #[allow(non_snake_case)]
        unsafe impl SCStreamOutput for ScribaAudioOutput {
            #[unsafe(method(stream:didOutputSampleBuffer:ofType:))]
            unsafe fn stream_didOutputSampleBuffer_ofType(
                &self,
                _stream: &SCStream,
                sample_buffer: &CMSampleBuffer,
                output_type: SCStreamOutputType,
            ) {
                // Only process audio buffers
                if output_type != SCStreamOutputType::Audio {
                    return;
                }
                // Extract audio data from the CMSampleBuffer
                if let Some(samples) = unsafe { extract_audio_samples(sample_buffer) } {
                    if let Ok(mut guard) = self.ivars().encoder.try_lock() {
                        if let Some(enc) = guard.as_mut() {
                            let _ = enc.encode_samples(&samples);
                        }
                    }
                }
            }
        }
    );

    impl ScribaAudioOutput {
        fn new(encoder: AudioEncoderHandle) -> Retained<Self> {
            let this = Self::alloc().set_ivars(AudioOutputIvars { encoder });
            unsafe { msg_send![super(this), init] }
        }
    }

    /// Extract f32 audio samples from a CMSampleBuffer.
    ///
    /// ScreenCaptureKit delivers audio as 32-bit float interleaved PCM.
    unsafe fn extract_audio_samples(sample_buffer: &CMSampleBuffer) -> Option<Vec<f32>> {
        // Get the block buffer containing the raw audio bytes
        let block_buffer = unsafe { sample_buffer.data_buffer()? };

        let mut total_length: usize = 0;
        let mut data_ptr: *mut i8 = std::ptr::null_mut();

        let status = unsafe {
            block_buffer.data_pointer(
                0,
                std::ptr::null_mut(), // length_at_offset (we don't need it)
                &mut total_length,
                &mut data_ptr,
            )
        };

        if status != 0 || data_ptr.is_null() || total_length == 0 {
            return None;
        }

        let bytes = unsafe { std::slice::from_raw_parts(data_ptr as *const u8, total_length) };
        Some(bytes_to_f32_samples(bytes))
    }

    /// Convert raw bytes to f32 PCM samples.
    fn bytes_to_f32_samples(data: &[u8]) -> Vec<f32> {
        if data.len() % 4 == 0 {
            data.chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()
        } else if data.len() % 2 == 0 {
            data.chunks_exact(2)
                .map(|chunk| {
                    let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                    sample as f32 / 32768.0
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Platform-specific inner handle for macOS.
    pub(super) struct PlatformLoopback {
        _stream: Retained<SCStream>,
        _output: Retained<ScribaAudioOutput>,
    }

    impl Drop for PlatformLoopback {
        fn drop(&mut self) {
            // Stop the capture stream
            let (tx, rx) = std::sync::mpsc::channel();
            let block = RcBlock::new(move |_error: *mut NSError| {
                let _ = tx.send(());
            });
            unsafe {
                self._stream
                    .stopCaptureWithCompletionHandler(Some(&block));
            }
            // Wait briefly for stop to complete
            let _ = rx.recv_timeout(std::time::Duration::from_secs(2));
        }
    }

    /// Start capturing system audio via ScreenCaptureKit.
    pub(super) fn start_capture(
        _device_hint: Option<&str>,
        output_path: &std::path::Path,
    ) -> Result<LoopbackSession> {
        let sample_rate: u32 = 48000;
        let channels: u16 = 2;

        // Create encoder for loopback audio
        let settings = CompressionSettings {
            format: AudioFormat::Wav,
            sample_rate,
            bitrate_kbps: None,
            channels,
            speech_optimized: false,
        };
        let encoder = create_encoder(output_path, &settings)?;
        let encoder: AudioEncoderHandle = Arc::new(Mutex::new(Some(encoder)));

        // Get shareable content (blocking)
        let content = get_shareable_content_sync()?;
        let displays: Retained<NSArray<SCDisplay>> =
            unsafe { content.displays() };

        if displays.count() == 0 {
            return Err(anyhow::anyhow!("No display found for ScreenCaptureKit"));
        }
        let display = &displays.objectAtIndex(0);

        // Create content filter for the display (captures all system audio)
        let empty_windows: Retained<NSArray<objc2_screen_capture_kit::SCWindow>> =
            NSArray::new();
        let filter = unsafe {
            SCContentFilter::initWithDisplay_excludingWindows(
                SCContentFilter::alloc(),
                display,
                &empty_windows,
            )
        };

        // Configure audio-only capture
        let config = unsafe {
            let c = SCStreamConfiguration::new();
            c.setCapturesAudio(true);
            c.setExcludesCurrentProcessAudio(false);
            c.setSampleRate(sample_rate as isize);
            c.setChannelCount(channels as isize);
            // Disable video capture to save resources
            c.setWidth(1);
            c.setHeight(1);
            c.setMinimumFrameInterval(objc2_core_media::CMTime {
                value: 1,
                timescale: 1,
                flags: objc2_core_media::CMTimeFlags(0),
                epoch: 0,
            });
            c
        };

        // Create the stream
        let stream = unsafe {
            SCStream::initWithFilter_configuration_delegate(
                SCStream::alloc(),
                &filter,
                &config,
                None, // No delegate needed for basic capture
            )
        };

        // Create and add the audio output handler
        let audio_output = ScribaAudioOutput::new(encoder.clone());
        unsafe {
            stream
                .addStreamOutput_type_sampleHandlerQueue_error(
                    ProtocolObject::from_ref(&*audio_output),
                    SCStreamOutputType::Audio,
                    None, // Use default queue
                )
                .map_err(|e| anyhow::anyhow!("Failed to add audio output: {}", e))?;
        }

        // Start capture (blocking on completion)
        let (tx, rx) = std::sync::mpsc::channel();
        let block = RcBlock::new(move |error: *mut NSError| {
            if error.is_null() {
                let _ = tx.send(Ok(()));
            } else {
                let desc = unsafe { (*error).localizedDescription() }.to_string();
                let _ = tx.send(Err(desc));
            }
        });
        unsafe {
            stream.startCaptureWithCompletionHandler(Some(&block));
        }
        match rx.recv_timeout(std::time::Duration::from_secs(5)) {
            Ok(Ok(())) => {}
            Ok(Err(desc)) => {
                return Err(anyhow::anyhow!(
                    "ScreenCaptureKit start failed: {}",
                    desc
                ))
            }
            Err(_) => {
                return Err(anyhow::anyhow!(
                    "ScreenCaptureKit start timed out"
                ))
            }
        }

        Ok(LoopbackSession {
            handle: LoopbackHandle {
                _inner: LoopbackInner::MacOS(PlatformLoopback {
                    _stream: stream,
                    _output: audio_output,
                }),
            },
            encoder,
            sample_rate,
            channels,
        })
    }

    /// Synchronously get shareable content via a blocking completion handler.
    fn get_shareable_content_sync() -> Result<Retained<SCShareableContent>> {
        let (tx, rx) = std::sync::mpsc::channel();
        let cell: Arc<UnsafeCell<Option<Retained<SCShareableContent>>>> =
            Arc::new(UnsafeCell::new(None));
        let cell_clone = cell.clone();

        let block = RcBlock::new(move |content: *mut SCShareableContent, error: *mut NSError| {
            if !content.is_null() && error.is_null() {
                unsafe {
                    let retained = Retained::retain(content).unwrap();
                    *cell_clone.get() = Some(retained);
                }
                let _ = tx.send(Ok(()));
            } else {
                let desc = if !error.is_null() {
                    unsafe { (*error).localizedDescription() }.to_string()
                } else {
                    "Unknown error".to_string()
                };
                let _ = tx.send(Err(desc));
            }
        });

        unsafe {
            SCShareableContent::getShareableContentWithCompletionHandler(&block);
        }

        match rx.recv_timeout(std::time::Duration::from_secs(5)) {
            Ok(Ok(())) => unsafe {
                (*cell.get())
                    .take()
                    .ok_or_else(|| anyhow::anyhow!("Failed to get shareable content"))
            },
            Ok(Err(desc)) => Err(anyhow::anyhow!(
                "ScreenCaptureKit content query failed: {}",
                desc
            )),
            Err(_) => Err(anyhow::anyhow!(
                "ScreenCaptureKit content query timed out"
            )),
        }
    }

    /// Detect available loopback sources on macOS.
    pub(super) fn detect_sources() -> Result<Vec<String>> {
        match get_shareable_content_sync() {
            Ok(content) => {
                let displays: Retained<NSArray<SCDisplay>> =
                    unsafe { content.displays() };
                let mut sources = Vec::new();
                for i in 0..displays.count() {
                    sources.push(format!("ScreenCaptureKit (Display {})", i + 1));
                }
                Ok(sources)
            }
            Err(_) => Ok(vec![]),
        }
    }
}

// ── Linux: cpal with PulseAudio/PipeWire monitor sources ─────────────────

#[cfg(target_os = "linux")]
mod platform {
    use super::*;
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    /// Known patterns for PulseAudio/PipeWire monitor sources.
    const MONITOR_PATTERNS: &[&str] = &["monitor of", ".monitor", "monitor"];

    /// Platform-specific inner handle for Linux.
    pub(super) struct PlatformLoopback {
        _stream: cpal::Stream,
    }

    impl Drop for PlatformLoopback {
        fn drop(&mut self) {
            // cpal::Stream stops on drop
        }
    }

    /// Start capturing system audio via a PulseAudio/PipeWire monitor source.
    pub(super) fn start_capture(
        device_hint: Option<&str>,
        output_path: &std::path::Path,
    ) -> Result<LoopbackSession> {
        let host = cpal::default_host();
        let device = find_monitor_device(&host, device_hint)?;
        let device_config = device
            .default_input_config()
            .context("Failed to get loopback device config")?;

        let sample_rate = device_config.sample_rate().0;
        let channels = device_config.channels();

        let settings = CompressionSettings {
            format: AudioFormat::Wav,
            sample_rate,
            bitrate_kbps: None,
            channels,
            speech_optimized: false,
        };
        let encoder = create_encoder(output_path, &settings)?;
        let encoder: AudioEncoderHandle = Arc::new(Mutex::new(Some(encoder)));

        let encoder_clone = encoder.clone();
        let stream = match device_config.sample_format() {
            cpal::SampleFormat::F32 => device.build_input_stream(
                &device_config.clone().into(),
                move |data: &[f32], _: &_| {
                    if let Ok(mut guard) = encoder_clone.try_lock() {
                        if let Some(enc) = guard.as_mut() {
                            let _ = enc.encode_samples(data);
                        }
                    }
                },
                |err| eprintln!("Loopback stream error: {}", err),
                None,
            )?,
            cpal::SampleFormat::I16 => {
                let enc = encoder_clone;
                device.build_input_stream(
                    &device_config.clone().into(),
                    move |data: &[i16], _: &_| {
                        let samples: Vec<f32> =
                            data.iter().map(|&s| s as f32 / 32768.0).collect();
                        if let Ok(mut guard) = enc.try_lock() {
                            if let Some(encoder) = guard.as_mut() {
                                let _ = encoder.encode_samples(&samples);
                            }
                        }
                    },
                    |err| eprintln!("Loopback stream error: {}", err),
                    None,
                )?
            }
            cpal::SampleFormat::I32 => {
                let enc = encoder_clone;
                device.build_input_stream(
                    &device_config.clone().into(),
                    move |data: &[i32], _: &_| {
                        let samples: Vec<f32> =
                            data.iter().map(|&s| s as f32 / 2147483648.0).collect();
                        if let Ok(mut guard) = enc.try_lock() {
                            if let Some(encoder) = guard.as_mut() {
                                let _ = encoder.encode_samples(&samples);
                            }
                        }
                    },
                    |err| eprintln!("Loopback stream error: {}", err),
                    None,
                )?
            }
            fmt => {
                return Err(anyhow::anyhow!(
                    "Unsupported loopback sample format: {:?}",
                    fmt
                ))
            }
        };

        stream.play().context("Failed to start loopback stream")?;

        Ok(LoopbackSession {
            handle: LoopbackHandle {
                _inner: LoopbackInner::Linux(PlatformLoopback { _stream: stream }),
            },
            encoder,
            sample_rate,
            channels,
        })
    }

    fn find_monitor_device(host: &cpal::Host, hint: Option<&str>) -> Result<cpal::Device> {
        let devices = host
            .input_devices()
            .context("Failed to enumerate input devices")?;

        if let Some(name) = hint {
            let name_lower = name.to_lowercase();
            for device in devices {
                if let Ok(dev_name) = device.name() {
                    if dev_name.to_lowercase().contains(&name_lower) {
                        return Ok(device);
                    }
                }
            }
            return Err(anyhow::anyhow!(
                "Loopback device '{}' not found. Use `scriba health --verbose` to list devices.",
                name
            ));
        }

        let devices = host
            .input_devices()
            .context("Failed to enumerate input devices")?;
        for device in devices {
            if let Ok(dev_name) = device.name() {
                let lower = dev_name.to_lowercase();
                if MONITOR_PATTERNS.iter().any(|pat| lower.contains(pat)) {
                    return Ok(device);
                }
            }
        }

        Err(anyhow::anyhow!(
            "No PulseAudio/PipeWire monitor source found. \
             Make sure PulseAudio or PipeWire is running."
        ))
    }

    pub(super) fn detect_sources() -> Result<Vec<String>> {
        let host = cpal::default_host();
        let devices = host
            .input_devices()
            .context("Failed to enumerate input devices")?;

        let mut sources = Vec::new();
        for device in devices {
            if let Ok(dev_name) = device.name() {
                let lower = dev_name.to_lowercase();
                if MONITOR_PATTERNS.iter().any(|pat| lower.contains(pat)) {
                    sources.push(dev_name);
                }
            }
        }
        Ok(sources)
    }
}

// ── Unsupported platforms ───────────────────────────────────────────────

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
mod platform {
    use super::*;

    pub(super) struct PlatformLoopback;

    pub(super) fn start_capture(
        _device_hint: Option<&str>,
        _output_path: &std::path::Path,
    ) -> Result<LoopbackSession> {
        Err(anyhow::anyhow!(
            "System audio loopback is not supported on this platform."
        ))
    }

    pub(super) fn detect_sources() -> Result<Vec<String>> {
        Ok(vec![])
    }
}

// ── Unified interface ────────────────────────────────────────────────────

/// Inner enum for platform-specific loopback handle.
#[allow(dead_code)]
enum LoopbackInner {
    #[cfg(target_os = "macos")]
    MacOS(platform::PlatformLoopback),
    #[cfg(target_os = "linux")]
    Linux(platform::PlatformLoopback),
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    Unsupported(platform::PlatformLoopback),
}

/// Start capturing system audio output.
///
/// - `device_hint`: Optional device name hint. On macOS this is ignored
///   (ScreenCaptureKit captures all system audio). On Linux this is used
///   to find a specific PulseAudio/PipeWire monitor source.
/// - `output_path`: Path to write the loopback WAV file.
///
/// Returns a `LoopbackSession` containing the handle (drop to stop),
/// encoder, and the capture sample rate/channels.
pub fn start_loopback_capture(
    device_hint: Option<&str>,
    output_path: &std::path::Path,
) -> Result<LoopbackSession> {
    platform::start_capture(device_hint, output_path)
}

/// Detect available loopback audio sources on the current platform.
///
/// On macOS: returns ScreenCaptureKit display sources.
/// On Linux: returns PulseAudio/PipeWire monitor sources.
pub fn detect_loopback_sources() -> Result<Vec<String>> {
    platform::detect_sources()
}
