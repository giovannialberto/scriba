class Scriba < Formula
  desc "Modern CLI tool for recording and transcribing audio using OpenAI Whisper"
  homepage "https://github.com/giovannialberto/scriba"
  version "0.28.4"
  
  if Hardware::CPU.intel?
    url "https://github.com/giovannialberto/scriba/releases/download/v0.28.4/scriba-x86_64-apple-darwin"
    sha256 "9714b3333512dd0c502ca493ae457da816144b018362563d6caacd9ba2d0786a"
  else
    url "https://github.com/giovannialberto/scriba/releases/download/v0.28.4/scriba-aarch64-apple-darwin"
    sha256 "4399ddaf239075bff0cd242197e3071ef329fcbd1af2093c19fd1ea226ca164e"
  end
  
  def install
    bin.install "scriba-#{Hardware::CPU.intel? ? "x86_64" : "aarch64"}-apple-darwin" => "scriba"
  end
  
  test do
    assert_match version.to_s, shell_output("#{bin}/scriba --version")
  end
end
