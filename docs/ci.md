# Tests and CI

The workflow is [`.github/workflows/ci.yml`](../.github/workflows/ci.yml). It
runs on every push to `main` and on every pull request.

## What it does

1. Installs the pinned dependencies (`requirements-dev.txt`) on Python 3.13.
2. Builds the Docker image for linux/amd64.
3. Runs all tests with `REQUIRE_CONTAINER=1`.

The tests cover features, splits, metrics, the registry export, the API
contract, and the container. The container test starts the real image and
checks that it predicts exactly like the training code.

## Publishing

After the tests pass on a push to `main`, a second job builds the image again
and pushes it to GitHub Container Registry as
`ghcr.io/mimmospoto/loan-default-scoring`, tagged `latest` and with the short
commit hash. Pull requests only run the tests. The job uses the built-in
`GITHUB_TOKEN`, so there is no password to manage.

## Design decisions

- **The exported model is committed.** `artifacts/model.joblib` (371 KB) and
  `artifacts/metadata.json` are the only files in `artifacts/` that git
  tracks. CI needs the model to build the image, and the files are small. The
  cost is that each new model version adds a binary to git history.
- **Data is never needed.** Training data is not in the repo, so no test reads
  it. API tests train a small model on random applications.
- **A missing image fails the build.** Locally the container test skips when
  Docker or the image is missing. In CI, `REQUIRE_CONTAINER=1` turns that skip
  into a failure, so the parity check cannot pass by not running.
- **Same pins as local work.** CI installs the same requirements files, so a
  green run means the pinned versions work together.
- **Read-only permissions.** The workflow only needs to read the repo.

## Limitations

- No linting or type checking yet.
- No vulnerability scan of the image.
- The publish job rebuilds the image instead of reusing the tested one. The
  inputs are identical, but a stricter setup would push the exact image the
  tests ran against.
- Deploying to Azure comes next (Phase 7).
