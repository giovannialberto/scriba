class Scriba < Formula
  desc "Modern CLI tool for recording and transcribing audio using OpenAI Whisper"
  homepage "https://github.com/giovannialberto/scriba"
  version "0.28.1"
  
  if Hardware::CPU.intel?
    url "https://github.com/giovannialberto/scriba/releases/download/v0.28.1/scriba-x86_64-apple-darwin"
    sha256 "c1504d3f525025f90727276617e55f7fdfa676a6b4d02af086a9fd66fcbf1505"
  else
    url "https://github.com/giovannialberto/scriba/releases/download/v0.28.1/scriba-aarch64-apple-darwin"
    sha256 "a406507b07a915253995607ee63db2d66bf89958cc6b25c8ca42ebf9273b9931"
  end
  
  def install
    bin.install "scriba-#{Hardware::CPU.intel? ? "x86_64" : "aarch64"}-apple-darwin" => "scriba"
  end
  
  test do
    assert_match version.to_s, shell_output("#{bin}/scriba --version")
  end
end
