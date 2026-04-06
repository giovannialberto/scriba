class Scriba < Formula
  desc "Modern CLI tool for recording and transcribing audio using OpenAI Whisper"
  homepage "https://github.com/giovannialberto/scriba"
  version "0.25.0"
  
  if Hardware::CPU.intel?
    url "https://github.com/giovannialberto/scriba/releases/download/v0.25.0/scriba-x86_64-apple-darwin"
    sha256 "a05ae3fbc6915b1a7ee77d050449c865527852f5b3c243ea8155fe3e7244edda"
  else
    url "https://github.com/giovannialberto/scriba/releases/download/v0.25.0/scriba-aarch64-apple-darwin"
    sha256 "5029558fbd9975b3a0cc4efbc19ee806dc29a78c8d4fd0ef43b207ba94b311e0"
  end
  
  def install
    bin.install "scriba-#{Hardware::CPU.intel? ? "x86_64" : "aarch64"}-apple-darwin" => "scriba"
  end
  
  test do
    assert_match version.to_s, shell_output("#{bin}/scriba --version")
  end
end
