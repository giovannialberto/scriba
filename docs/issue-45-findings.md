# Issue #45 — Findings: sherpa-onnx Migration & Model Benchmarks

## Summary

We replaced `whisper-rs` (whisper.cpp bindings) with [`sherpa-onnx`](https://github.com/k2-fsa/sherpa-onnx) as Scriba's transcription backend. This gives us access to multiple STT model families through a single dependency, with Silero VAD for long-audio segmentation.

### What changed
- **Removed**: `whisper-rs`, `pyannote-rs`, `ort`, `num_cpus`
- **Added**: `sherpa-onnx` (official Rust crate, v1.12), `gag` (stderr suppression)
- **Diarization**: Replaced `pyannote-rs` with sherpa-onnx's built-in offline speaker diarization
- **VAD**: Silero VAD segments long audio incrementally (no 30-second Whisper window limit)
- **Models available**: Whisper (Tiny→Turbo via ONNX), SenseVoice, Parakeet TDT 0.6B v3

Config migration is seamless — old `"model_size": "Medium"` auto-deserializes to the new format via serde aliases.

---

## Benchmark: 14.8 min Italian/English recording (Apple Silicon M-series)

| Model | Total Time* | Transcription Speed | Italian Quality | English Quality |
|-------|------------|-------------------|-----------------|-----------------|
| **Whisper Turbo** (int8) | 268s | 1x baseline | Poor — translates Italian to broken English, repetitive artifacts | Acceptable |
| **SenseVoice** (int8) | 74s | **3.6x faster** | Unusable — detects wrong languages (ja/ko/zh) | Passable when English-only |
| **Parakeet TDT 0.6B v3** (int8) | 102s | **2.6x faster** | Excellent — preserves Italian words, proper nouns, natural code-switching | Good |

*Total time includes enrichment (Ollama LLM call ~25-30s constant overhead). Pure transcription time difference is even larger.

### Quality comparison (same segment)

**Whisper Turbo:**
> ti spiego un azzimino come funziona il GitHub e le cose che tv ml and the repo where they are - It's a core service of forecasting. that every morning you take the meteorite, the historical data, the forecast

**Parakeet TDT:**
> ti spiego un attimino come funzionano il GitHub le cose che ci sono. T V M L. E la repo dove ci stanno? Il servizio core di forecasting. Quello che ogni mattina si prende i dati meteo, si prende i dati storici, fai forcast

**SenseVoice:**
> let speak know turn不 wasの。 一不め。 エラレーボルシさんの。 Its a bitォカテンうん。

Parakeet TDT is the clear winner for Italian: faster than Whisper *and* dramatically better quality. SenseVoice is designed for zh/en/ja/ko/yue and fails on Italian.

---

## CoreML / Apple Neural Engine

We investigated CoreML acceleration on macOS. The prebuilt static sherpa-onnx binaries bundle an onnxruntime **without** the CoreML execution provider. Switching to shared linking enables the CoreML EP, but ONNX Runtime's CoreML path doesn't fully leverage the Apple Neural Engine the way WhisperKit does (WhisperKit uses pre-compiled `.mlmodelc` models optimized for ANE). We did not observe meaningful speedup through the ONNX Runtime CoreML EP and reverted to CPU-only static linking.

The 8-10x speedup @m13v reported is specific to WhisperKit's native CoreML pipeline, not achievable through ONNX Runtime's CoreML execution provider.

---

## Phase 1 status

- [x] Replace `whisper-rs` with `sherpa-onnx` as transcription backend
- [x] Replace `pyannote-rs` with sherpa-onnx diarization
- [x] Silero VAD for long-audio segmentation (no 30s limit)
- [x] Whisper ONNX models (Tiny through Turbo)
- [x] SenseVoice support
- [x] Parakeet TDT 0.6B v3 support
- [x] Config migration for existing users
- [x] Onboarding flow updated with new models
- [x] Model auto-download on first use

## Next steps (Phase 2+)

- [ ] Evaluate Parakeet TDT on more languages and recording conditions
- [ ] Investigate streaming transcription with Zipformer for real-time use
- [ ] Explore building sherpa-onnx from source with CoreML for true ANE acceleration
- [ ] Consider removing SenseVoice from the model list (or gating it for zh/en/ja/ko only)
