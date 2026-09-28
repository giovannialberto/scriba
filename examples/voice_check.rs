//! Voice-profile check on one recording. Not part of `cargo test`.
//!
//! ```text
//! cargo build --example voice_check
//! target/debug/examples/voice_check <audio> [verbose_json_with_words.json] [min_secs]
//! ```
//! Cuts the audio into turns (host word timing when given, Silero VAD
//! otherwise), embeds each turn, and prints its similarity to the owner's
//! stored voice profile plus the pairwise similarity between turns.

use std::path::Path;

use scriba::core::diarization::{
    DiarizationModels, SpeakerEmbedder, TimedWord, TranscriptSegment, WORD_TURN_GAP_SECS,
    read_wav_channels, turns_from_words,
};
use scriba::database::Database;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let audio = Path::new(&args[1]);
    let min_secs: f32 = args.get(3).and_then(|a| a.parse().ok()).unwrap_or(0.8);
    let home = std::env::var("HOME").unwrap();

    let wav = std::env::temp_dir().join("voice_check_16k.wav");
    let ok = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(audio)
        .args(["-ar", "16000", "-ac", "1", "-f", "wav"])
        .arg(&wav)
        .status()
        .unwrap()
        .success();
    assert!(ok, "ffmpeg failed");
    let (channels, rate) = read_wav_channels(&wav).unwrap();
    let audio_samples = &channels[0];

    let turns: Vec<TranscriptSegment> = if let (Some(path), Ok(_)) = (args.get(2), std::env::var("SCRIBA_SEGMENTS")) {
        // A plain segments array ({start, end, text}) used as the units.
        serde_json::from_str::<Vec<TranscriptSegment>>(&std::fs::read_to_string(path).unwrap()).unwrap()
    } else if let Some(path) = args.get(2) {
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let words: Vec<TimedWord> = v["words"]
            .as_array()
            .unwrap()
            .iter()
            .map(|w| TimedWord {
                start: w["start"].as_f64().unwrap() as f32,
                end: w["end"].as_f64().unwrap() as f32,
                text: w["word"].as_str().unwrap().to_string(),
            })
            .collect();
        turns_from_words(&words, WORD_TURN_GAP_SECS)
    } else {
        let mut cfg = sherpa_onnx::VadModelConfig::default();
        cfg.silero_vad.model = Some(format!("{home}/scriba_recordings/models/sherpa/silero_vad.onnx"));
        cfg.silero_vad.threshold = 0.5;
        cfg.silero_vad.min_silence_duration = 0.5;
        cfg.silero_vad.min_speech_duration = 0.25;
        cfg.silero_vad.window_size = 512;
        cfg.sample_rate = 16000;
        cfg.num_threads = 1;
        let vad = sherpa_onnx::VoiceActivityDetector::create(&cfg, 30.0).expect("vad");
        let mut out = Vec::new();
        let drain = |vad: &sherpa_onnx::VoiceActivityDetector, out: &mut Vec<TranscriptSegment>| {
            while !vad.is_empty() {
                if let Some(seg) = vad.front() {
                    let start = seg.start() as f32 / rate as f32;
                    out.push(TranscriptSegment {
                        start,
                        end: start + seg.samples().len() as f32 / rate as f32,
                        text: String::new(),
                    });
                }
                vad.pop();
            }
        };
        for chunk in audio_samples.chunks(512) {
            vad.accept_waveform(chunk);
            drain(&vad, &mut out);
        }
        vad.flush();
        drain(&vad, &mut out);
        out
    };

    let models = DiarizationModels {
        embedding: match std::env::var("SCRIBA_SPK_MODEL") {
            Ok(p) => Path::new(&p).to_path_buf(),
            Err(_) => Path::new(&home).join(
                "scriba_recordings/models/sherpa/nemo_en_titanet_large.onnx",
            ),
        },
    };
    let embedder = SpeakerEmbedder::load(&models).expect("model");
    let profiles = Database::new().and_then(|db| db.speaker_profiles(scriba::core::diarization::EMBEDDING_MODEL_ID)).unwrap_or_default();
    let owner = profiles.iter().find(|p| p.is_owner);
    match owner {
        Some(p) => println!("owner profile: {} samples, {:.0}s", p.samples, p.total_secs),
        None => println!("no owner profile"),
    }

    let mut embs: Vec<Option<Vec<f32>>> = Vec::new();
    for t in &turns {
        let s = (t.start * rate as f32) as usize;
        let e = ((t.end * rate as f32) as usize).min(audio_samples.len());
        let dur = t.end - t.start;
        let emb = if dur >= min_secs && e > s {
            // Same as SpeakerEmbedder::embed but with the caller's minimum length.
            let mut padded = audio_samples[s..e].to_vec();
            while (padded.len() as f32) < 0.8 * rate as f32 {
                padded.extend_from_slice(&audio_samples[s..e]);
            }
            embedder.embed(&padded, rate)
        } else {
            None
        };
        let sim = match (&emb, owner) {
            (Some(e), Some(p)) if p.centroid.len() == e.len() => format!("{:.2}", p.similarity(e)),
            _ => "  - ".to_string(),
        };
        println!(
            "{:6.2}-{:6.2} ({:4.1}s) owner={} {}",
            t.start, t.end, dur, sim, t.text
        );
        embs.push(emb);
    }

    println!("pairwise:");
    let idx: Vec<usize> = (0..embs.len()).filter(|&i| embs[i].is_some()).collect();
    print!("      ");
    for &j in &idx {
        print!("{:>5}", j);
    }
    println!();
    for &i in &idx {
        print!("{:>5} ", i);
        for &j in &idx {
            let a = embs[i].as_ref().unwrap();
            let b = embs[j].as_ref().unwrap();
            let c: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            print!("{:>5.2}", c);
        }
        println!();
    }
}
