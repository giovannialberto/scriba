class Scriba < Formula
  desc "Modern CLI tool for recording and transcribing audio using OpenAI Whisper"
  homepage "https://github.com/giovannialberto/scriba"
  version "0.28.0"
  
  if Hardware::CPU.intel?
    url "https://github.com/giovannialberto/scriba/releases/download/v0.28.0/scriba-x86_64-apple-darwin"
    sha256 "3348840f7cc9ccdad53aa59d330c31a7dfaabb0c5e9a142086506f3bb9d178ab"
  else
    url "https://github.com/giovannialberto/scriba/releases/download/v0.28.0/scriba-aarch64-apple-darwin"
    sha256 "3fd238e90ae8c4a097255aaefc44f14683250dee6567f96bd362b41e045539fb"
  end
  
  def install
    bin.install "scriba-#{Hardware::CPU.intel? ? "x86_64" : "aarch64"}-apple-darwin" => "scriba"
  end
  
  test do
    assert_match version.to_s, shell_output("#{bin}/scriba --version")
  end
end
