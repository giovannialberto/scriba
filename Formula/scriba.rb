class Scriba < Formula
  desc "Modern CLI tool for recording and transcribing audio using OpenAI Whisper"
  homepage "https://github.com/giovannialberto/scriba"
  version "0.25.1"
  
  if Hardware::CPU.intel?
    url "https://github.com/giovannialberto/scriba/releases/download/v0.25.1/scriba-x86_64-apple-darwin"
    sha256 "72f6ef071111c10b6fcd9b5eedd6a7c1522b247b4c44a3f3c8fa7e7a992b62a4"
  else
    url "https://github.com/giovannialberto/scriba/releases/download/v0.25.1/scriba-aarch64-apple-darwin"
    sha256 "31257daec86f17590561cced51af3b46641ac31d34d8f61cfe176c2835822e43"
  end
  
  def install
    bin.install "scriba-#{Hardware::CPU.intel? ? "x86_64" : "aarch64"}-apple-darwin" => "scriba"
  end
  
  test do
    assert_match version.to_s, shell_output("#{bin}/scriba --version")
  end
end
