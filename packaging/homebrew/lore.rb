# Homebrew formula — copy into a tap repo at:
#   github.com/ReiPano/homebrew-lore/Formula/lore.rb
#
# After publishing a PyPI release, update `url` and `sha256` to the new
# artifact. The GitHub Actions release workflow uploads sdist + wheel to the
# tagged release, so you can compute the SHA from either asset.
#
#   shasum -a 256 lore_memory-0.5.0-py3-none-any.whl

class Lore < Formula
  include Language::Python::Virtualenv

  desc "Local hybrid search (BM25 + vector) with MCP and HTTP interfaces"
  homepage "https://github.com/ReiPano/lore"
  url "https://files.pythonhosted.org/packages/source/l/lore-memory/lore_memory-0.5.0.tar.gz"
  sha256 "REPLACE_WITH_SHA256"
  license "MIT"

  depends_on "python@3.12"
  depends_on "docker" => :recommended

  def install
    virtualenv_install_with_resources
  end

  def caveats
    <<~EOS
      Lore stores data under ~/.lore. Start Qdrant first:

        lore init
        lore up --watch
        lore projects add ~/your/project

      Register with Claude Code / Claude Desktop per README:
        https://github.com/ReiPano/lore#claude-code--claude-desktop-mcp
    EOS
  end

  test do
    system bin/"lore", "--help"
  end
end
