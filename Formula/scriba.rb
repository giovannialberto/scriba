class Scriba < Formula
  desc "Modern CLI tool for recording and transcribing audio using OpenAI Whisper"
  homepage "https://github.com/giovannialberto/scriba"
  version "0.22.3"
  
  if Hardware::CPU.intel?
    url "https://github.com/giovannialberto/scriba/releases/download/v0.22.3/scriba-x86_64-apple-darwin"
    sha256 "db5f1ba162208f23b96d742e8dbca898fe00361d35b16a9260dc085364a08602"
  else
    url "https://github.com/giovannialberto/scriba/releases/download/v0.22.3/scriba-aarch64-apple-darwin"
    sha256 "a9651093536682ad3069d884987e0395c07dd0c8746bd9697312fd00db77b3d6"
  end
  
  def install
    bin.install "scriba-#{Hardware::CPU.intel? ? "x86_64" : "aarch64"}-apple-darwin" => "scriba"
  end
  
  test do
    assert_match version.to_s, shell_output("#{bin}/scriba --version")
  end
end
