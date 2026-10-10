"""API contract: what a caller sends and what comes back.

Field names are exactly MODEL_FEATURES. Ranges are wider than anything seen in
training (see docs/api.md), so they reject typos and impossible values, not
unusual applicants. Optional fields are the ones that are genuinely missing for
some training loans; the model reads a missing value as "not available".
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_BATCH = 1000

Purpose = Literal[
    "car",
    "credit_card",
    "debt_consolidation",
    "home_improvement",
    "house",
    "major_purchase",
    "medical",
    "moving",
    "other",
    "renewable_energy",
    "small_business",
    "vacation",
    "wedding",
]
HomeOwnership = Literal["RENT", "MORTGAGE", "OWN", "OTHER"]
VerificationStatus = Literal["Not Verified", "Source Verified", "Verified"]


class LoanApplication(BaseModel):
    """One loan application, with only what is known at application time."""

    # A misspelled optional field would silently count as missing, so unknown names are errors.
    model_config = ConfigDict(extra="forbid")

    purpose: Purpose
    home_ownership: HomeOwnership
    verification_status: VerificationStatus
    loan_amnt: float = Field(ge=500, le=100_000, description="Requested amount, USD")
    annual_inc: float = Field(ge=0, le=10_000_000, description="Self-reported annual income, USD")
    dti: float = Field(ge=0, le=100, description="Monthly debt payments over monthly income, in percent")
    fico: float = Field(ge=300, le=850, description="FICO score at application")
    credit_history_months: float = Field(ge=0, le=1200, description="Months since the earliest credit line")
    delinq_2yrs: float = Field(ge=0, le=100, description="Delinquencies in the past 2 years")
    inq_last_6mths: float = Field(ge=0, le=50, description="Credit inquiries in the past 6 months")
    open_acc: float = Field(ge=0, le=150, description="Open credit lines")
    total_acc: float = Field(ge=0, le=300, description="Credit lines ever opened")
    pub_rec: float = Field(ge=0, le=100, description="Derogatory public records")
    pub_rec_bankruptcies: float = Field(ge=0, le=50, description="Public record bankruptcies")
    revol_bal: float = Field(ge=0, le=10_000_000, description="Total revolving balance, USD")
    mort_acc: float = Field(ge=0, le=100, description="Mortgage accounts")
    acc_open_past_24mths: float = Field(ge=0, le=100, description="Accounts opened in the past 24 months")
    emp_length_years: float | None = Field(
        default=None, ge=0, le=10, description="Years employed, 10 means 10 or more; omit if unknown"
    )
    mths_since_last_delinq: float | None = Field(
        default=None, ge=0, le=1200, description="Months since the last delinquency; omit if there is none"
    )
    revol_util: float | None = Field(
        default=None, ge=0, le=1000, description="Revolving balance over credit limit, in percent; omit if unknown"
    )

    @model_validator(mode="after")
    def open_accounts_are_part_of_total(self) -> LoanApplication:
        if self.open_acc > self.total_acc:
            raise ValueError("open_acc cannot exceed total_acc")
        return self


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    applications: list[LoanApplication] = Field(min_length=1, max_length=MAX_BATCH)


class PredictResponse(BaseModel):
    pd: float = Field(description="Calibrated probability of default over the life of a 36-month loan")
    risk_band: Literal["A", "B", "C", "D", "E"] = Field(description="A is the lowest risk, E the highest")
    model_version: str
    request_id: str


class Prediction(BaseModel):
    pd: float
    risk_band: Literal["A", "B", "C", "D", "E"]


class BatchResponse(BaseModel):
    predictions: list[Prediction] = Field(description="One per application, in the order sent")
    model_version: str
    request_id: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_version: str
