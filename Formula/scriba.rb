class Scriba < Formula
  desc "Modern CLI tool for recording and transcribing audio using OpenAI Whisper"
  homepage "https://github.com/giovannialberto/scriba"
  version "0.30.0"
  
  if Hardware::CPU.intel?
    url "https://github.com/giovannialberto/scriba/releases/download/v0.30.0/scriba-x86_64-apple-darwin"
    sha256 "a4d54b9620cd8d3546ad9a949655fc135cf011ff73c40eb5e9ffa145a5651c1f"
  else
    url "https://github.com/giovannialberto/scriba/releases/download/v0.30.0/scriba-aarch64-apple-darwin"
    sha256 "b3ef3d0db70f7bfe92ce5c070e729dc6f2f8cad6a5bf45f6bf96dbc3aeff2e41"
  end
  
  def install
    bin.install "scriba-#{Hardware::CPU.intel? ? "x86_64" : "aarch64"}-apple-darwin" => "scriba"
  end
  
  test do
    assert_match version.to_s, shell_output("#{bin}/scriba --version")
  end
end
