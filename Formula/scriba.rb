class Scriba < Formula
  desc "Modern CLI tool for recording and transcribing audio using OpenAI Whisper"
  homepage "https://github.com/giovannialberto/scriba"
  version "0.26.1"
  
  if Hardware::CPU.intel?
    url "https://github.com/giovannialberto/scriba/releases/download/v0.26.1/scriba-x86_64-apple-darwin"
    sha256 "463568a4bbf5d0de9d061851a0124d9f7ca836febade873403079736847517c7"
  else
    url "https://github.com/giovannialberto/scriba/releases/download/v0.26.1/scriba-aarch64-apple-darwin"
    sha256 "273d7360c470dc41a3507d9999050dac387a9b8e01abeaab77bec82c899be0f9"
  end
  
  def install
    bin.install "scriba-#{Hardware::CPU.intel? ? "x86_64" : "aarch64"}-apple-darwin" => "scriba"
  end
  
  test do
    assert_match version.to_s, shell_output("#{bin}/scriba --version")
  end
end
