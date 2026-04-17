class Scriba < Formula
  desc "Modern CLI tool for recording and transcribing audio using OpenAI Whisper"
  homepage "https://github.com/giovannialberto/scriba"
  version "0.27.0"
  
  if Hardware::CPU.intel?
    url "https://github.com/giovannialberto/scriba/releases/download/v0.27.0/scriba-x86_64-apple-darwin"
    sha256 "bd0283e5a7a9a9d35ddcb52b8c8979af434af33610ca195c031bf2373b2fbda8"
  else
    url "https://github.com/giovannialberto/scriba/releases/download/v0.27.0/scriba-aarch64-apple-darwin"
    sha256 "bbd9f24de3b3f2208b9e003c11402eca3baa422520bade1cd098fe68bae8c429"
  end
  
  def install
    bin.install "scriba-#{Hardware::CPU.intel? ? "x86_64" : "aarch64"}-apple-darwin" => "scriba"
  end
  
  test do
    assert_match version.to_s, shell_output("#{bin}/scriba --version")
  end
end
