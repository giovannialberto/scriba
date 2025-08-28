# Scriba

Modern CLI to record audio and transcribe it locally using Whisper via whisper-rs (no API keys, fully offline).

## ✨ Features

- 🎙️ **Audio Recording** - High-quality microphone recording with smart MP3 compression
- 📝 **Local Transcription** - Whisper.cpp engine via whisper-rs with progress indicators  
- 📊 **Interactive Dashboard** - Browse, search, and manage recordings with live statistics
- 🔍 **Full-Text Search** - Find recordings by searching transcript content
- ▶️ **Audio Playback** - Play recordings directly from the dashboard
- 📋 **Transcript Management** - Copy, view, and transcribe recordings with one keystroke

## 🚀 Quick Start

### Installation
```bash
# Homebrew (recommended)
brew tap giovannialberto/scriba
brew install scriba

# From source
cargo install --git https://github.com/giovannialberto/scriba
```

### Whisper Model
Download a Whisper ggml model (e.g., ggml-base.en.bin) and place it under `~/scriba_recordings/models/`:
```
mkdir -p ~/scriba_recordings/models
# Place model file here, e.g.:
# ~/scriba_recordings/models/ggml-base.en.bin
```

### Usage
```bash
# Launch interactive dashboard
scriba

# Record and transcribe in CLI (choose model)
scriba record --model turbo
scriba transcribe audio-file.wav --model large
```

## 📊 Dashboard Controls

| Key | Action |
|-----|--------|
| **R** | Record + Auto-transcribe |
| **A** | Record audio only |
| **T** | Transcribe selected recording |
| **Enter** | View transcript |
| **C** | Copy transcript to clipboard |
| **P** | Play recording |
| **/** | Search transcripts |
| **D** | Delete recording |
| **H** | Show help |

## 🗂️ File Organization

All recordings are stored in `~/scriba_recordings/`:
```
~/scriba_recordings/
├── 2025-08-26_14-30-25_meeting/
│   ├── recording.mp3
│   └── transcript.txt
└── scriba.db
```

## 🔧 Requirements

- **Whisper model** - ggml file under `~/scriba_recordings/models/` (e.g., `ggml-base.en.bin`)
- **CMake** - Required to build whisper-rs (`brew install cmake` on macOS)
- **FFmpeg** - For audio compression and resampling (`brew install ffmpeg`)
- **Audio system** - Microphone for recording, speakers for playback

Model selection
- Use `--model tiny|base|small|medium|large|turbo`.
- Dashboard default: `turbo` (auto-downloads on first run).
- CLI default: `medium` (override with `--model`).
- Turbo uses a GGUF build by default (`ggml-large-v3-turbo-q5_0.gguf`). You can override by placing your own model file under `~/scriba_recordings/models/` (e.g., `ggml-large-v3-turbo.gguf`).
- First use auto-downloads the selected model to `~/scriba_recordings/models/`.

## 🎯 Key Benefits

- **80-90% file size reduction** with speech-optimized MP3 compression
- **Instant transcription** with animated progress indicators
- **Seamless workflow** - record, transcribe, and manage from one interface
- **Smart search** - find any recording by searching transcript text
- **Cross-platform** - works on macOS, Linux, and Windows

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.
