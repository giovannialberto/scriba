class Scriba < Formula
  desc "Modern CLI tool for recording and transcribing audio using OpenAI Whisper"
  homepage "https://github.com/giovannialberto/scriba"
  version "0.28.3"
  
  if Hardware::CPU.intel?
    url "https://github.com/giovannialberto/scriba/releases/download/v0.28.3/scriba-x86_64-apple-darwin"
    sha256 "cf90e2c17c89d1f8ecf64af6f0c78b066d0bb916b9e93417aa951fcfc93705e7"
  else
    url "https://github.com/giovannialberto/scriba/releases/download/v0.28.3/scriba-aarch64-apple-darwin"
    sha256 "3a4fbecbff6cdd5045775a0177582a63720e3958029baa807e7be7284fd4e83a"
  end
  
  def install
    bin.install "scriba-#{Hardware::CPU.intel? ? "x86_64" : "aarch64"}-apple-darwin" => "scriba"
  end
  
  test do
    assert_match version.to_s, shell_output("#{bin}/scriba --version")
  end
end
