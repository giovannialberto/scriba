class Scriba < Formula
  desc "Modern CLI tool for recording and transcribing audio using OpenAI Whisper"
  homepage "https://github.com/giovannialberto/scriba"
  version "0.24.1"
  
  if Hardware::CPU.intel?
    url "https://github.com/giovannialberto/scriba/releases/download/v0.24.1/scriba-x86_64-apple-darwin"
    sha256 "7bf95ebfd9de33bf08ee6e5f65944646f1f5d83e04fa8d8892bd40cda17aaf6b"
  else
    url "https://github.com/giovannialberto/scriba/releases/download/v0.24.1/scriba-aarch64-apple-darwin"
    sha256 "1919a38947f159d3244435225c50475ada0a4c08a05f63607789523914aa2463"
  end
  
  def install
    bin.install "scriba-#{Hardware::CPU.intel? ? "x86_64" : "aarch64"}-apple-darwin" => "scriba"
  end
  
  test do
    assert_match version.to_s, shell_output("#{bin}/scriba --version")
  end
end
