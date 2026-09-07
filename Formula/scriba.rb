class Scriba < Formula
  desc "Modern CLI tool for recording and transcribing audio using OpenAI Whisper"
  homepage "https://github.com/giovannialberto/scriba"
  version "0.28.2"
  
  if Hardware::CPU.intel?
    url "https://github.com/giovannialberto/scriba/releases/download/v0.28.2/scriba-x86_64-apple-darwin"
    sha256 "4f7de7e47209f4aa7dc5432c1b27708d0abe138efe8c807207b229eb9183c9d0"
  else
    url "https://github.com/giovannialberto/scriba/releases/download/v0.28.2/scriba-aarch64-apple-darwin"
    sha256 "7cd331940cf4a808f9c0a7b3d715fd6c60891331fa4dd946cf3d85a2139e88bc"
  end
  
  def install
    bin.install "scriba-#{Hardware::CPU.intel? ? "x86_64" : "aarch64"}-apple-darwin" => "scriba"
  end
  
  test do
    assert_match version.to_s, shell_output("#{bin}/scriba --version")
  end
end
