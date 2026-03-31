class Scriba < Formula
  desc "Modern CLI tool for recording and transcribing audio using OpenAI Whisper"
  homepage "https://github.com/giovannialberto/scriba"
  version "0.23.1"
  
  if Hardware::CPU.intel?
    url "https://github.com/giovannialberto/scriba/releases/download/v0.23.1/scriba-x86_64-apple-darwin"
    sha256 "ab1a79816c4d392b77d4f50d2db55a348f4b0465c6f8727c9551b2e46d0389ae"
  else
    url "https://github.com/giovannialberto/scriba/releases/download/v0.23.1/scriba-aarch64-apple-darwin"
    sha256 "e9bff4f4b65ba590392eab3ebea8a2d2e20170dfa38f245a5c7c4169daec0f2f"
  end
  
  def install
    bin.install "scriba-#{Hardware::CPU.intel? ? "x86_64" : "aarch64"}-apple-darwin" => "scriba"
  end
  
  test do
    assert_match version.to_s, shell_output("#{bin}/scriba --version")
  end
end
