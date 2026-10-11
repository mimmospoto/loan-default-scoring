# Docker

The image serves the API with the exported model inside. See the
[Dockerfile](../Dockerfile).

## Build and run

```bash
.venv/bin/python -m loan_default.registry export        # once: writes artifacts/
docker build --platform linux/amd64 -t loan-default:dev .
docker run --rm -p 8000:8000 loan-default:dev            # http://127.0.0.1:8000/docs
```

## Design decisions

- **Two stages.** The builder installs the pinned packages into a virtual
  environment. The final image copies only that environment and the code, so
  pip caches and build files stay out.
- **Pinned dependencies.** `requirements.txt` holds the serving packages only,
  with exact versions. Training and registry packages (MLflow, pyarrow) are in
  `requirements-train.txt`, and tests and Kaggle are in `requirements-dev.txt`.
  The image has no MLflow.
- **`libgomp1`.** LightGBM needs the OpenMP runtime, which the slim base image
  does not include.
- **linux/amd64.** This is what Azure Container Apps runs, and it matches the
  Intel Mac used to build.
- **The model is baked in.** The image copies `model.joblib` and
  `metadata.json`, so one image is one model version. The cost is a rebuild for
  each new model. Mounting `artifacts/` at run time would avoid that, but the
  deployed container would then depend on a volume.
- **Non-root user,** and a `HEALTHCHECK` that calls `/health`.
- **`.dockerignore` allows only what is needed.** Data, `mlruns/`, notebooks
  and secrets can never end up in the image.

## Same predictions as training

`tests/test_container.py` starts the image, sends 200 applications to
`/predict/batch`, and compares the result with the training code. The
probabilities must be identical and the bands equal. The test is skipped if
Docker, the image or an exported model is missing.

## Size

The image is about 690 MB on disk (158 MB compressed). Most of it is
scikit-learn, pandas, LightGBM and their dependencies.

## Limitations

- The model file is git-ignored, so CI must get it from somewhere before it can
  build the image (Phase 6).
- Images are not scanned for vulnerabilities yet.
