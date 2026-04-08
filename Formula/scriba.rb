class Scriba < Formula
  desc "Modern CLI tool for recording and transcribing audio using OpenAI Whisper"
  homepage "https://github.com/giovannialberto/scriba"
  version "0.26.0"
  
  if Hardware::CPU.intel?
    url "https://github.com/giovannialberto/scriba/releases/download/v0.26.0/scriba-x86_64-apple-darwin"
    sha256 "630364224b6cd1a477d7837a79fcfb5bd77d03b74f80fff621b8f2690bb85233"
  else
    url "https://github.com/giovannialberto/scriba/releases/download/v0.26.0/scriba-aarch64-apple-darwin"
    sha256 "3aad45db63121906c6ad3de21b0974de56093754ccb74976d2a3bcef66e06e67"
  end
  
  def install
    bin.install "scriba-#{Hardware::CPU.intel? ? "x86_64" : "aarch64"}-apple-darwin" => "scriba"
  end
  
  test do
    assert_match version.to_s, shell_output("#{bin}/scriba --version")
  end
end
