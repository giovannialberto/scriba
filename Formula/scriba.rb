class Scriba < Formula
  desc "Modern CLI tool for recording and transcribing audio using OpenAI Whisper"
  homepage "https://github.com/giovannialberto/scriba"
  version "0.22.0"
  
  if Hardware::CPU.intel?
    url "https://github.com/giovannialberto/scriba/releases/download/v0.22.0/scriba-x86_64-apple-darwin"
    sha256 "248333f0572b9980e8d2a7297bd943ba351331cbdff9cf854c69bb5b4f384f9c"
  else
    url "https://github.com/giovannialberto/scriba/releases/download/v0.22.0/scriba-aarch64-apple-darwin"
    sha256 "b7aea5106de1ee35fa2f90f3dfeca803392a1bbe838a85d1302ff8ea8855e883"
  end
  
  def install
    bin.install "scriba-#{Hardware::CPU.intel? ? "x86_64" : "aarch64"}-apple-darwin" => "scriba"
  end
  
  test do
    assert_match version.to_s, shell_output("#{bin}/scriba --version")
  end
end
