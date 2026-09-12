//! Calibration bench for speaker clustering. Not part of `cargo test`.
//!
//! ```text
//! cargo build --example diarize_bench
//! target/debug/examples/diarize_bench <audio.(mp3|wav)> <segments.json> [threshold ...]
//! ```
//! `segments.json` is an array of `{start, end, text, speaker?}`; when a
//! `speaker` field is present it is treated as ground truth and purity is
//! reported. Uses the model in ~/scriba_recordings/models/sherpa.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use scriba::core::diarization::{
    DiarizationModels, DiarizationOptions, SpeakerEmbedder, TranscriptSegment, cluster_speakers,
    fill_gaps_by_neighbour, read_wav_channels,
};

#[derive(serde::Deserialize)]
struct Seg {
    start: f32,
    end: f32,
    #[serde(default)]
    text: String,
    #[serde(default)]
    speaker: Option<String>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let audio = Path::new(&args[1]);
    let segs: Vec<Seg> = if args[2] == "vad" {
        Vec::new()
    } else {
        serde_json::from_str(&std::fs::read_to_string(&args[2]).unwrap()).unwrap()
    };
    let thresholds: Vec<f32> = if args.len() > 3 {
        args[3..].iter().map(|a| a.parse().unwrap()).collect()
    } else {
        vec![0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    };

    let wav = std::env::temp_dir().join("diarize_bench_16k.wav");
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
    let segs: Vec<Seg> = if segs.is_empty() {
        // Segment with Silero VAD exactly like the transcriber does.
        let home = std::env::var("HOME").unwrap();
        let mut cfg = sherpa_onnx::VadModelConfig::default();
        cfg.silero_vad.model = Some(format!(
            "{home}/scriba_recordings/models/sherpa/silero_vad.onnx"
        ));
        cfg.silero_vad.threshold = 0.5;
        cfg.silero_vad.min_silence_duration = 0.5;
        cfg.silero_vad.min_speech_duration = 0.25;
        cfg.silero_vad.window_size = 512;
        cfg.sample_rate = 16000;
        cfg.num_threads = 1;
        let vad = sherpa_onnx::VoiceActivityDetector::create(&cfg, 30.0).expect("vad");
        let mut out = Vec::new();
        let drain = |vad: &sherpa_onnx::VoiceActivityDetector, out: &mut Vec<Seg>| {
            while !vad.is_empty() {
                if let Some(seg) = vad.front() {
                    let start = seg.start() as f32 / rate as f32;
                    out.push(Seg {
                        start,
                        end: start + seg.samples().len() as f32 / rate as f32,
                        text: String::new(),
                        speaker: None,
                    });
                }
                vad.pop();
            }
        };
        for chunk in channels[0].chunks(512) {
            vad.accept_waveform(chunk);
            drain(&vad, &mut out);
        }
        vad.flush();
        drain(&vad, &mut out);
        println!("VAD found {} segments", out.len());
        out
    } else {
        segs
    };

    let home = std::env::var("HOME").unwrap();
    let models = DiarizationModels {
        embedding: match std::env::var("SCRIBA_SPK_MODEL") {
            Ok(p) => Path::new(&p).to_path_buf(),
            Err(_) => Path::new(&home).join("scriba_recordings/models/sherpa/nemo_en_titanet_large.onnx"),
        },
    };
    let embedder = SpeakerEmbedder::load(&models).expect("model");
    let transcript: Vec<TranscriptSegment> = segs
        .iter()
        .map(|s| TranscriptSegment {
            start: s.start,
            end: s.end,
            text: s.text.clone(),
        })
        .collect();
    let durations: Vec<f32> = transcript.iter().map(|t| t.end - t.start).collect();

    let started = Instant::now();
    let embeddings = embedder.embed_segments(&channels[0], rate, &transcript);
    let embedded = embeddings.iter().filter(|e| e.is_some()).count();
    println!(
        "{} segments, {} embedded, {:.0}s of speech, embeddings in {:.1}s",
        transcript.len(),
        embedded,
        durations.iter().sum::<f32>(),
        started.elapsed().as_secs_f64()
    );

    let truth: Vec<Option<&str>> = segs.iter().map(|s| s.speaker.as_deref()).collect();
    let has_truth = truth.iter().any(|t| t.is_some());
    for th in thresholds {
        let started = Instant::now();
        let mut assignment = cluster_speakers(
            &embeddings,
            &durations,
            DiarizationOptions {
                similarity_threshold: th,
                max_speakers: 8,
            },
        );
        fill_gaps_by_neighbour(&mut assignment);
        let mut sizes: HashMap<usize, f32> = HashMap::new();
        for (a, d) in assignment.iter().zip(&durations) {
            if let Some(c) = a {
                *sizes.entry(*c).or_default() += d;
            }
        }
        let mut sorted: Vec<f32> = sizes.values().copied().collect();
        sorted.sort_by(|a, b| b.total_cmp(a));
        let top: Vec<String> = sorted.iter().take(6).map(|s| format!("{s:.0}s")).collect();
        let mut line = format!(
            "threshold {th:.2}: {} speakers, speech per speaker: {}  ({:.2}s)",
            sizes.len(),
            top.join(" "),
            started.elapsed().as_secs_f64()
        );
        if has_truth {
            // Purity: for each cluster, the share of its duration belonging to its majority true speaker.
            let mut per_cluster: HashMap<usize, HashMap<&str, f32>> = HashMap::new();
            for ((a, d), t) in assignment.iter().zip(&durations).zip(&truth) {
                if let (Some(c), Some(t)) = (a, t) {
                    *per_cluster.entry(*c).or_default().entry(t).or_default() += d;
                }
            }
            let total: f32 = durations.iter().sum();
            let pure: f32 = per_cluster
                .values()
                .map(|m| m.values().copied().fold(0.0, f32::max))
                .sum();
            let true_speakers = truth
                .iter()
                .flatten()
                .collect::<std::collections::HashSet<_>>()
                .len();
            line.push_str(&format!(
                "  purity {:.0}% (truth: {} speakers)",
                100.0 * pure / total,
                true_speakers
            ));
        }
        println!("{line}");
    }
}
