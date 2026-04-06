#!/bin/sh
set -e

REPO="giovannialberto/scriba"
INSTALL_DIR="/usr/local/bin"

# Detect platform
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
  Darwin)
    case "$ARCH" in
      x86_64) TARGET="x86_64-apple-darwin" ;;
      arm64)  TARGET="aarch64-apple-darwin" ;;
      *)      echo "Unsupported architecture: $ARCH"; exit 1 ;;
    esac
    ;;
  Linux)
    case "$ARCH" in
      x86_64) TARGET="x86_64-unknown-linux-gnu" ;;
      *)      echo "Unsupported architecture: $ARCH (pre-built binaries are x86_64 only)"; exit 1 ;;
    esac
    ;;
  *)
    echo "Unsupported OS: $OS"; exit 1 ;;
esac

# Get latest release tag
LATEST="$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" | grep '"tag_name"' | cut -d'"' -f4)"
if [ -z "$LATEST" ]; then
  echo "Failed to fetch latest release"; exit 1
fi

URL="https://github.com/$REPO/releases/download/$LATEST/scriba-$TARGET"
echo "Installing scriba $LATEST for $TARGET..."

TMPFILE="$(mktemp)"
curl -fsSL "$URL" -o "$TMPFILE"
chmod +x "$TMPFILE"

if [ -w "$INSTALL_DIR" ]; then
  mv "$TMPFILE" "$INSTALL_DIR/scriba"
else
  echo "Installing to $INSTALL_DIR (requires sudo)..."
  sudo mv "$TMPFILE" "$INSTALL_DIR/scriba"
fi

echo "scriba installed to $INSTALL_DIR/scriba"
scriba --version
