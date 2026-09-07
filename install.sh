#!/bin/sh
# Scriba installer — macOS (Apple Silicon + Intel) and Linux (x86_64).
#
#   curl -fsSL https://raw.githubusercontent.com/giovannialberto/scriba/main/install.sh | sh
#
# Installs the latest release binary into ~/.local/bin (no sudo), verifies its
# SHA-256 against the published checksum, and prints a PATH hint if needed.
#
# Environment overrides:
#   SCRIBA_INSTALL_DIR   target directory        (default: ~/.local/bin)
#   SCRIBA_VERSION       release tag to install  (default: latest, e.g. v0.28.2)
set -eu

REPO="giovannialberto/scriba"
INSTALL_DIR="${SCRIBA_INSTALL_DIR:-$HOME/.local/bin}"

say()  { printf '%s\n' "$*"; }
fail() { printf 'error: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || fail "'$1' is required but not installed"; }

need curl
need uname

# ── Platform ────────────────────────────────────────────────────────────────
OS="$(uname -s)"
ARCH="$(uname -m)"
case "$OS" in
  Darwin)
    case "$ARCH" in
      arm64)  TARGET="aarch64-apple-darwin" ;;
      x86_64) TARGET="x86_64-apple-darwin" ;;
      *)      fail "unsupported macOS architecture: $ARCH" ;;
    esac ;;
  Linux)
    case "$ARCH" in
      x86_64|amd64) TARGET="x86_64-unknown-linux-gnu" ;;
      *) fail "unsupported Linux architecture: $ARCH (pre-built binaries are x86_64 only; build from source with cargo)" ;;
    esac ;;
  *) fail "unsupported OS: $OS" ;;
esac

# ── Version ─────────────────────────────────────────────────────────────────
VERSION="${SCRIBA_VERSION:-}"
if [ -z "$VERSION" ]; then
  VERSION="$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
    | grep '"tag_name"' | head -n1 | cut -d'"' -f4)"
  [ -n "$VERSION" ] || fail "could not determine the latest release (GitHub API unreachable?)"
fi
case "$VERSION" in v*) ;; *) VERSION="v$VERSION" ;; esac

BASE="https://github.com/$REPO/releases/download/$VERSION"
ASSET="scriba-$TARGET"

# ── Download + verify ───────────────────────────────────────────────────────
TMPDIR_="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_"' EXIT

say "Downloading scriba $VERSION ($TARGET)..."
curl -fsSL "$BASE/$ASSET" -o "$TMPDIR_/scriba" || fail "download failed: $BASE/$ASSET"
curl -fsSL "$BASE/$ASSET.sha256" -o "$TMPDIR_/scriba.sha256" || fail "checksum download failed"

EXPECTED="$(cut -d' ' -f1 < "$TMPDIR_/scriba.sha256")"
if command -v sha256sum >/dev/null 2>&1; then
  ACTUAL="$(sha256sum "$TMPDIR_/scriba" | cut -d' ' -f1)"
else
  ACTUAL="$(shasum -a 256 "$TMPDIR_/scriba" | cut -d' ' -f1)"
fi
[ "$EXPECTED" = "$ACTUAL" ] || fail "checksum mismatch (expected $EXPECTED, got $ACTUAL)"
chmod +x "$TMPDIR_/scriba"

# ── Install ─────────────────────────────────────────────────────────────────
mkdir -p "$INSTALL_DIR" || fail "cannot create $INSTALL_DIR"
[ -w "$INSTALL_DIR" ] || fail "$INSTALL_DIR is not writable (set SCRIBA_INSTALL_DIR to another directory)"
mv "$TMPDIR_/scriba" "$INSTALL_DIR/scriba"
# Downloads via curl aren't quarantined, but clear the flag defensively on macOS.
if [ "$OS" = "Darwin" ] && command -v xattr >/dev/null 2>&1; then
  xattr -d com.apple.quarantine "$INSTALL_DIR/scriba" 2>/dev/null || true
fi

say "Installed scriba $VERSION to $INSTALL_DIR/scriba"

# ── Post-install hints ──────────────────────────────────────────────────────
case ":$PATH:" in
  *":$INSTALL_DIR:"*) ;;
  *)
    say ""
    say "Add $INSTALL_DIR to your PATH, e.g. for zsh:"
    say "  echo 'export PATH=\"$INSTALL_DIR:\$PATH\"' >> ~/.zshrc && source ~/.zshrc"
    ;;
esac

if ! command -v ffmpeg >/dev/null 2>&1; then
  say ""
  say "Scriba needs FFmpeg for audio encoding:"
  case "$OS" in
    Darwin) say "  brew install ffmpeg" ;;
    Linux)  say "  sudo apt install ffmpeg   # or: dnf install ffmpeg / pacman -S ffmpeg" ;;
  esac
fi

say ""
say "Run 'scriba' to get started."
