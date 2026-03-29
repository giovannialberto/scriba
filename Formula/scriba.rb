class Scriba < Formula
  desc "Modern CLI tool for recording and transcribing audio using OpenAI Whisper"
  homepage "https://github.com/giovannialberto/scriba"
  version "0.22.1"
  
  if Hardware::CPU.intel?
    url "https://github.com/giovannialberto/scriba/releases/download/v0.22.1/scriba-x86_64-apple-darwin"
    sha256 "1b649e19b4cefdb27a822e82e7260cfa48737cb7b435196523c8784869c85ec4"
  else
    url "https://github.com/giovannialberto/scriba/releases/download/v0.22.1/scriba-aarch64-apple-darwin"
    sha256 "da9b152be9aa80777913ed3c8f4d13031a08a2a63f51dd0d7180ac4e2234eae4"
  end
  
  def install
    bin.install "scriba-#{Hardware::CPU.intel? ? "x86_64" : "aarch64"}-apple-darwin" => "scriba"
  end
  
  test do
    assert_match version.to_s, shell_output("#{bin}/scriba --version")
  end
end
