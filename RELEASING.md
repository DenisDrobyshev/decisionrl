# Releasing

The package is distributed on PyPI as **`reinforce-rl`** (the import name stays
`reinforce`). Publishing is automated: pushing a **GitHub Release** triggers
[`.github/workflows/publish.yml`](.github/workflows/publish.yml), which builds the
sdist + wheel and uploads them via **PyPI Trusted Publishing** (OIDC — no API
token or secret stored in the repo).

## One-time setup (required before the first release)

Trusted publishing has to be authorized on PyPI *before* the first upload, while
the project name is still unclaimed ("pending publisher"):

1. Sign in at <https://pypi.org> → **Your projects** → **Publishing** →
   **Add a pending publisher**.
2. Fill in exactly:
   - **PyPI Project Name:** `reinforce-rl`
   - **Owner:** `DenisDrobyshev`
   - **Repository name:** `reinforce`
   - **Workflow name:** `publish.yml`
   - **Environment name:** `pypi`
3. Save. (The GitHub side already declares `environment: pypi` and
   `permissions: id-token: write`, so no further repo config is needed.)

## Cutting a release

1. Bump `version` in [`pyproject.toml`](pyproject.toml) and move the
   `## [Unreleased]` section of [`CHANGELOG.md`](CHANGELOG.md) under the new
   version + date.
2. Commit and tag:
   ```bash
   git commit -am "release: v0.1.0"
   git tag v0.1.0
   git push origin main --tags
   ```
3. On GitHub: **Releases → Draft a new release**, pick the tag, publish.
4. The **Publish to PyPI** workflow runs automatically. When it goes green:
   ```bash
   pip install reinforce-rl
   ```

## Verifying the build locally

```bash
python -m build            # -> dist/reinforce_rl-<ver>-py3-none-any.whl + .tar.gz
python -m twine check dist/*   # metadata sanity check (should PASS)
```

> Do **not** publish manually with `twine upload` for routine releases — let the
> GitHub Release + trusted publisher flow do it so provenance is attached.
