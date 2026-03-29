//! Native audio playback via rodio.
//!
//! Replaces subprocess-based playback (mpv, ffplay, afplay) with
//! in-process audio decoding and playback. Supports WAV, MP3, OGG,
//! FLAC and other formats via symphonia (included by rodio).

use anyhow::{Context, Result};
use rodio::Decoder;
use rodio::stream::{DeviceSinkBuilder, MixerDeviceSink};
use rodio::Player;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// In-process audio player backed by rodio.
///
/// Plays audio files natively without requiring external tools.
/// Drop to stop playback, or call `stop()` explicitly.
pub struct AudioPlayer {
    player: Player,
    // Keep the device handle alive for the duration of playback.
    _handle: MixerDeviceSink,
}

impl AudioPlayer {
    /// Start playing an audio file. Supports WAV, MP3, OGG, FLAC.
    ///
    /// Playback runs on a background thread managed by rodio.
    /// Returns immediately; use `is_playing()` to check status.
    pub fn play_file(path: &Path) -> Result<Self> {
        let mut handle = DeviceSinkBuilder::open_default_sink()
            .context("Failed to open audio output device")?;
        handle.log_on_drop(false);

        let file = File::open(path)
            .with_context(|| format!("Failed to open audio file: {}", path.display()))?;
        let reader = BufReader::new(file);
        let source = Decoder::new(reader)
            .with_context(|| format!("Failed to decode audio file: {}", path.display()))?;

        let player = Player::connect_new(&handle.mixer());
        player.append(source);

        Ok(Self {
            player,
            _handle: handle,
        })
    }

    /// Stop playback immediately.
    pub fn stop(&self) {
        self.player.stop();
    }

    /// Check if audio is still playing.
    pub fn is_playing(&self) -> bool {
        !self.player.empty()
    }

    /// Set playback volume (0.0 = silent, 1.0 = full).
    pub fn set_volume(&self, volume: f32) {
        self.player.set_volume(volume);
    }
}
