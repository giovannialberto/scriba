class Scriba < Formula
  desc "Modern CLI tool for recording and transcribing audio using OpenAI Whisper"
  homepage "https://github.com/giovannialberto/scriba"
  version "0.30.1"
  
  if Hardware::CPU.intel?
    url "https://github.com/giovannialberto/scriba/releases/download/v0.30.1/scriba-x86_64-apple-darwin"
    sha256 "311383d38b3010762bc4923dc58b864ff89e4604827cb57ce89b6f77d479c631"
  else
    url "https://github.com/giovannialberto/scriba/releases/download/v0.30.1/scriba-aarch64-apple-darwin"
    sha256 "8046b345d76e058d2c178c6e2280c2297c4f6d583885bed2f4191d2477623c1d"
  end
  
  def install
    bin.install "scriba-#{Hardware::CPU.intel? ? "x86_64" : "aarch64"}-apple-darwin" => "scriba"
  end
  
  test do
    assert_match version.to_s, shell_output("#{bin}/scriba --version")
  end
end
