"""The Docker image must give the same predictions as the training code.

Needs Docker, a built image and an exported model; otherwise it is skipped.
CI sets REQUIRE_CONTAINER=1 so a missing image fails instead of skipping.
Build first: docker build -t loan-default:dev .   (override the name with IMAGE)
"""

import os
import shutil
import subprocess
import time

import httpx
import numpy as np
import pytest

from loan_default.bands import assign_band
from loan_default.config import ARTIFACTS_DIR
from loan_default.model import predict_pd
from loan_default.service import Scorer
from tests.test_api import make_applications, training_style_frame

IMAGE = os.environ.get("IMAGE", "loan-default:dev")
PORT = 8765


def docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0


REQUIRED = bool(os.environ.get("REQUIRE_CONTAINER"))
pytestmark = [
    pytest.mark.skipif(not REQUIRED and not docker_ready(), reason=f"Docker image {IMAGE} not available"),
    pytest.mark.skipif(not REQUIRED and not (ARTIFACTS_DIR / "model.joblib").exists(), reason="no exported model"),
]


@pytest.fixture(scope="module")
def container():
    run = subprocess.run(
        ["docker", "run", "-d", "--rm", "-p", f"{PORT}:8000", IMAGE], capture_output=True, text=True, check=True
    )
    container_id = run.stdout.strip()
    try:
        for _ in range(60):
            try:
                if httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.5)
        else:
            pytest.fail("container did not become healthy")
        yield f"http://127.0.0.1:{PORT}"
    finally:
        subprocess.run(["docker", "stop", container_id], capture_output=True)


def test_container_predicts_exactly_like_the_training_code(container):
    scorer = Scorer.load(ARTIFACTS_DIR)
    applications = make_applications(200, seed=11)

    expected = predict_pd(scorer.model, training_style_frame(applications))
    response = httpx.post(f"{container}/predict/batch", json={"applications": applications}, timeout=30)

    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == scorer.version
    np.testing.assert_array_equal([p["pd"] for p in body["predictions"]], expected)
    assert [p["risk_band"] for p in body["predictions"]] == list(assign_band(expected, scorer.thresholds))
