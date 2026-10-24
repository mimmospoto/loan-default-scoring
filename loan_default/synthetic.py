"""Made-up loan applications, for tests and for the drift self-check.

Plausible values inside the API's ranges, not a model of real borrowers.
"""

from __future__ import annotations

from typing import get_args

import numpy as np

from loan_default.schemas import HomeOwnership, Purpose, VerificationStatus


def make_applications(n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    applications = []
    for _ in range(n):
        total = int(rng.integers(2, 60))
        application = {
            "purpose": str(rng.choice(get_args(Purpose))),
            "home_ownership": str(rng.choice(get_args(HomeOwnership))),
            "verification_status": str(rng.choice(get_args(VerificationStatus))),
            "loan_amnt": float(rng.integers(1000, 35000)),
            "annual_inc": float(rng.integers(15000, 200000)),
            "dti": float(rng.uniform(0, 40)),
            "fico": float(rng.integers(660, 840)),
            "credit_history_months": float(rng.integers(36, 500)),
            "delinq_2yrs": float(rng.integers(0, 4)),
            "inq_last_6mths": float(rng.integers(0, 5)),
            "open_acc": float(rng.integers(1, total + 1)),
            "total_acc": float(total),
            "pub_rec": float(rng.integers(0, 3)),
            "pub_rec_bankruptcies": float(rng.integers(0, 2)),
            "revol_bal": float(rng.integers(0, 60000)),
            "mort_acc": float(rng.integers(0, 5)),
            "acc_open_past_24mths": float(rng.integers(0, 10)),
        }
        # Leave each optional field out about a third of the time.
        for name, value in {
            "emp_length_years": float(rng.integers(0, 11)),
            "mths_since_last_delinq": float(rng.integers(0, 120)),
            "revol_util": float(rng.uniform(0, 100)),
        }.items():
            if rng.random() > 0.33:
                application[name] = value
        applications.append(application)
    return applications
