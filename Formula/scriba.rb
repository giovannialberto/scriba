class Scriba < Formula
  desc "Modern CLI tool for recording and transcribing audio using OpenAI Whisper"
  homepage "https://github.com/giovannialberto/scriba"
  version "0.26.2"
  
  if Hardware::CPU.intel?
    url "https://github.com/giovannialberto/scriba/releases/download/v0.26.2/scriba-x86_64-apple-darwin"
    sha256 "2c1e20a2df308141ea12dc8a189ae5ec89525045ff8ea742d6ca34d6ee318f77"
  else
    url "https://github.com/giovannialberto/scriba/releases/download/v0.26.2/scriba-aarch64-apple-darwin"
    sha256 "351fa6fe55af64585c7c517dfac670b06d80c78235ec417d86ba1ffab9c4a6d8"
  end
  
  def install
    bin.install "scriba-#{Hardware::CPU.intel? ? "x86_64" : "aarch64"}-apple-darwin" => "scriba"
  end
  
  test do
    assert_match version.to_s, shell_output("#{bin}/scriba --version")
  end
end
