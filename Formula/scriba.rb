class Scriba < Formula
  desc "Modern CLI tool for recording and transcribing audio using OpenAI Whisper"
  homepage "https://github.com/giovannialberto/scriba"
  version "0.31.0"
  
  if Hardware::CPU.intel?
    url "https://github.com/giovannialberto/scriba/releases/download/v0.31.0/scriba-x86_64-apple-darwin"
    sha256 "f2628c321fe69c6e460694b2f165fc7be14e2e357cd972886d0f14f08d69eb57"
  else
    url "https://github.com/giovannialberto/scriba/releases/download/v0.31.0/scriba-aarch64-apple-darwin"
    sha256 "260372664db7c8f52baaada71e4a83cf74041f42eb6a4784b6058eec3e2f57a6"
  end
  
  def install
    bin.install "scriba-#{Hardware::CPU.intel? ? "x86_64" : "aarch64"}-apple-darwin" => "scriba"
  end
  
  test do
    assert_match version.to_s, shell_output("#{bin}/scriba --version")
  end
end
