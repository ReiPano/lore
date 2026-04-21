# Homebrew tap

Distribute Lore via Homebrew so macOS users can `brew install lore` without
cloning this repo.

## One-time tap setup

1. Create an empty repo at `https://github.com/ReiPano/homebrew-lore`.
2. Commit [`lore.rb`](lore.rb) to that repo as `Formula/lore.rb`.

## Per-release steps

1. Push a new git tag (e.g. `v0.5.0`) in this repo. `.github/workflows/release.yml` builds the sdist + wheel and uploads them to both PyPI and the GitHub release.
2. Grab the sdist SHA-256:

   ```bash
   shasum -a 256 lore_memory-0.5.0.tar.gz
   ```

3. In the tap repo, bump `Formula/lore.rb`:

   ```ruby
   url "https://files.pythonhosted.org/packages/source/l/lore-memory/lore_memory-0.5.0.tar.gz"
   sha256 "<value from step 2>"
   ```

4. Commit and push. End users install with:

   ```bash
   brew install reipano/lore/lore
   ```

## Automation (optional)

The release workflow can be extended to run `brew bump-formula-pr` against the tap. That requires a PAT with `repo` scope stored as `HOMEBREW_GITHUB_API_TOKEN` in this repo's secrets. Keep it manual while the formula settles.
