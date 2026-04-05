class Scriba < Formula
  desc "Modern CLI tool for recording and transcribing audio using OpenAI Whisper"
  homepage "https://github.com/giovannialberto/scriba"
  version "0.24.0"
  
  if Hardware::CPU.intel?
    url "https://github.com/giovannialberto/scriba/releases/download/v0.24.0/scriba-x86_64-apple-darwin"
    sha256 "4f2b90b9d62907afd13e7d208c0fde65d89be6337f478917ec006b455ae1bd26"
  else
    url "https://github.com/giovannialberto/scriba/releases/download/v0.24.0/scriba-aarch64-apple-darwin"
    sha256 "7851f66fe439a4ea2b8d39e8f69719dcb7b78b56745f10d49da03414680f778e"
  end
  
  def install
    bin.install "scriba-#{Hardware::CPU.intel? ? "x86_64" : "aarch64"}-apple-darwin" => "scriba"
  end
  
  test do
    assert_match version.to_s, shell_output("#{bin}/scriba --version")
  end
end
