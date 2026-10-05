class Scriba < Formula
  desc "Modern CLI tool for recording and transcribing audio using OpenAI Whisper"
  homepage "https://github.com/giovannialberto/scriba"
  version "0.32.0"
  
  if Hardware::CPU.intel?
    url "https://github.com/giovannialberto/scriba/releases/download/v0.32.0/scriba-x86_64-apple-darwin"
    sha256 "ca483c3f70a44438473158f7a8ce1852b7152e24e45dc22914a1d8d841d3dde9"
  else
    url "https://github.com/giovannialberto/scriba/releases/download/v0.32.0/scriba-aarch64-apple-darwin"
    sha256 "c3ff0b32b2b7839582c50400e4fb434e69c7f76f9c56b3fbd6e88e555925d40a"
  end
  
  def install
    bin.install "scriba-#{Hardware::CPU.intel? ? "x86_64" : "aarch64"}-apple-darwin" => "scriba"
  end
  
  test do
    assert_match version.to_s, shell_output("#{bin}/scriba --version")
  end
end
