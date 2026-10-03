"""Download the Lending Club dataset from Kaggle into data/raw/.

Needs a Kaggle API token at ~/.kaggle/kaggle.json (or KAGGLE_USERNAME and
KAGGLE_KEY in the environment). Run: python -m loan_default.data.download
"""

from __future__ import annotations

import sys

from loan_default.config import KAGGLE_DATASET, RAW_DIR, find_accepted_file


def main() -> int:
    existing = find_accepted_file()
    if existing is not None:
        print(f"Already downloaded: {existing}")
        return 0

    # Importing kaggle authenticates immediately, so keep it inside main().
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
    except ImportError:
        print("kaggle is not installed; run: pip install -r requirements-dev.txt", file=sys.stderr)
        return 1
    except (Exception, SystemExit) as exc:
        print(f"Kaggle authentication failed: {exc}", file=sys.stderr)
        print("Put your API token at ~/.kaggle/kaggle.json (chmod 600).", file=sys.stderr)
        return 1

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {KAGGLE_DATASET} to {RAW_DIR} (about 1.3 GB zipped)...")
    api.dataset_download_files(KAGGLE_DATASET, path=str(RAW_DIR), unzip=True, quiet=False)

    downloaded = find_accepted_file()
    if downloaded is None:
        print(f"Download finished but no accepted-loans file found in {RAW_DIR}.", file=sys.stderr)
        return 1
    print(f"Ready: {downloaded}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
