# SOMA-X Release Checklist

This checklist covers public `py-soma-x` package releases from the public-safe
GitHub mirror.

## One-time PyPI setup

Configure Trusted Publishing for both PyPI and TestPyPI before creating a
release tag.

Use these publisher settings for the PyPI project `py-soma-x`:

- Owner: `NVlabs`
- Repository: `SOMA-X`
- Workflow: `pypi.yml`
- Environment: `pypi`

Use the same settings on TestPyPI with environment `testpypi`.

Protect the `pypi` and `testpypi` GitHub environments so publishing requires
maintainer approval. Do not configure long-lived PyPI API tokens for this
workflow.

## One-time Hugging Face setup

The v0.2.3 and later releases use Hugging Face Trusted Publishing. A maintainer
with Write access to `nvidia/SOMA-X` must add these claims under the repository's
**Settings -> Trusted Publishers** page:

- Provider: GitHub Actions
- Repository: `NVlabs/SOMA-X`
- Workflow: `pypi.yml`
- Branch: unset, because releases run from immutable version tags

Protect the `huggingface` GitHub environment so the first production sync
requires maintainer approval. The workflow requests a short-lived,
repository-scoped token through OIDC; do not configure a long-lived `HF_TOKEN`.

After initial setup and after every change to Hugging Face authentication or
publication code, run `pypi.yml` manually with `hf_git_smoke` enabled. The
smoke job uses the configured Trusted Publisher to create, verify, and remove
an ephemeral `ci-oidc-*` tag on the existing Hub snapshot. It does not upload
assets, publish packages, or require a package-version bump.

## Release steps

1. Merge the public-release prep MRs into internal `main`.
2. Cut or refresh the minor-line internal release branch, for example
   `release-0.2`. Patch releases reuse the same minor-line branch and create a
   new patch tag, for example `v0.2.1` from `release-0.2`.
3. Confirm `setup.cfg` and `soma/__init__.py` both contain the intended package
   version, for example `0.2.0`.
4. Post the exact public file diff, exact Hugging Face publish set, and proposed
   public-facing GitHub commit message to the release issue.
5. Obtain explicit human approval for all three review items before creating
   the public GitHub commit. Any subsequent change to those inputs requires a
   new review.
6. Mirror the public-safe release branch to public GitHub using the approved
   commit subject and body.
7. Confirm the generated public mirror candidate passed the internal
   public-release validation gate before pushing.
8. Confirm the public GitHub Actions build job passes on the release branch.
9. Run the `pypi.yml` `hf_git_smoke` workflow-dispatch preflight and confirm
   that the ephemeral Hub tag is created, verified, and removed. This is
   mandatory after an automation change and before the first release that uses
   the changed automation.
10. Create the release tag from the public-safe release branch:

   ```bash
   git tag -a vX.Y.Z -m "SOMA-X X.Y.Z"
   git push origin vX.Y.Z
   ```

11. Approve the protected `huggingface` environment, then verify the workflow
   publishes and immutably tags the exact manifest on `nvidia/SOMA-X`.
   The job repeats the ephemeral-tag preflight before uploading the release
   snapshot, so a Git authentication failure cannot occur after a large upload.
12. Verify the workflow downloads the Hub tag and passes file-set and SHA-256
   checks before the TestPyPI job becomes eligible.
13. Verify the tag-triggered workflow publishes to TestPyPI.
14. Approve the protected `pypi` environment only after TestPyPI verification.
15. Verify PyPI shows the new `py-soma-x` release.
16. Record release links for the GitHub tag, Hub tag/manifest, PyPI release,
    docs, and validation artifact.

## Interrupted Hugging Face release recovery

If the exact release snapshot reached Hub `main` but its immutable tag was not
created, do not bump the package version. Fix and smoke-test the automation,
then dispatch `pypi.yml` from public `main` with:

- `tag`: the existing public GitHub release tag, such as `v0.2.3`.
- `hf_revision`: the exact already-published Hub commit SHA.
- `hf_git_smoke`: disabled.

The recovery job uses the same short-lived OIDC credential to preflight Git
authentication, verify the Hub manifest at `hf_revision`, create the missing
release tag, and resume Hub verification, TestPyPI, PyPI, and the GitHub
Release. No personal Hugging Face token is required.

## Local checks

Run these from the release branch before tagging:

```bash
python tools/ci/check_release_version.py --expected vX.Y.Z
python -m build --sdist --wheel
python -m twine check --strict dist/*
```
