class Scriba < Formula
  desc "Modern CLI tool for recording and transcribing audio using OpenAI Whisper"
  homepage "https://github.com/giovannialberto/scriba"
  version "0.23.0"
  
  if Hardware::CPU.intel?
    url "https://github.com/giovannialberto/scriba/releases/download/v0.23.0/scriba-x86_64-apple-darwin"
    sha256 "1328e46c6bbe0649bb0e10736a70514bea019a039bdf3170f8f5c74a640a79a5"
  else
    url "https://github.com/giovannialberto/scriba/releases/download/v0.23.0/scriba-aarch64-apple-darwin"
    sha256 "79068f94f437b3ebbe9c1d1641cc1e24d6e8127decc7da7e3da599ec5e6fcd99"
  end
  
  def install
    bin.install "scriba-#{Hardware::CPU.intel? ? "x86_64" : "aarch64"}-apple-darwin" => "scriba"
  end
  
  test do
    assert_match version.to_s, shell_output("#{bin}/scriba --version")
  end
end
