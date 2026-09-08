class Scriba < Formula
  desc "Modern CLI tool for recording and transcribing audio using OpenAI Whisper"
  homepage "https://github.com/giovannialberto/scriba"
  version "0.29.0"
  
  if Hardware::CPU.intel?
    url "https://github.com/giovannialberto/scriba/releases/download/v0.29.0/scriba-x86_64-apple-darwin"
    sha256 "d4f0164fc28723698895c9d122f57ff6f2ef1f3b52761405f0e4937279f914e5"
  else
    url "https://github.com/giovannialberto/scriba/releases/download/v0.29.0/scriba-aarch64-apple-darwin"
    sha256 "8f76994da4a26515d91e31b19ab81bc5fd8a48e25e82d412e7dc519059cd9bc5"
  end
  
  def install
    bin.install "scriba-#{Hardware::CPU.intel? ? "x86_64" : "aarch64"}-apple-darwin" => "scriba"
  end
  
  test do
    assert_match version.to_s, shell_output("#{bin}/scriba --version")
  end
end
