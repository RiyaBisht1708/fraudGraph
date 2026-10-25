from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

from . import config, metrics
from .inference import FraudScorer
from .redis_store import RedisStore

state = {}


class Transaction(BaseModel):
    TransactionAmt: float
    ProductCD: str
    card1: float | int | str | None = None
    addr1: float | int | None = None
    DeviceInfo: str | None = None
    TransactionDT: float
    TransactionID: str | int | None = None
    # optional: pre-materialized feature vector from the feature store.
    # When absent, the scorer median-imputes the non-core fields.
    feature_vector: list[float] | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["scorer"] = None
    state["redis"] = RedisStore()
    state["startup_error"] = None
    try:
        scorer = FraudScorer()
        state["scorer"] = scorer
        metrics.configure(scorer.explainer.amt_mean, scorer.explainer.amt_std)
    except FileNotFoundError as exc:
        state["startup_error"] = str(exc)
        metrics.configure(0.0, 1.0)
    yield
    state.clear()


app = FastAPI(title="FraudGraph Scoring API", lifespan=lifespan)


def score_transaction(payload: dict) -> dict:
    store: RedisStore = state["redis"]
    scorer: FraudScorer | None = state["scorer"]
    if scorer is None:
        raise HTTPException(
            status_code=503,
            detail=(state.get("startup_error") or "FraudGraph model artifacts are missing. "
                    "Run the Kaggle notebook stage to produce artifacts/ before scoring."),
        )

    card1 = payload.get("card1")
    velocity = store.record_and_get_velocity(card1, float(payload["TransactionDT"]),
                                             float(payload["TransactionAmt"]))
    result = scorer.score(payload, velocity, feature_vector=payload.get("feature_vector"))
    if payload.get("TransactionID") is not None:
        store.push_neighbor(card1, payload["TransactionID"])
    metrics.observe(result["fraud_score"], result["decision"],
                    result["latency_ms"], float(payload["TransactionAmt"]))
    return result


@app.get("/health")
def health():
    redis_ok = state["redis"].ping() if "redis" in state else False
    return {
        "status": "ok" if state.get("scorer") is not None else "degraded",
        "redis": redis_ok,
        "model_ready": state.get("scorer") is not None,
        "startup_error": state.get("startup_error"),
    }


@app.get("/metrics")
def prometheus_metrics():
    body, content_type = metrics.render()
    return Response(content=body, media_type=content_type)


@app.post("/score")
def score(txn: Transaction):
    return score_transaction(txn.model_dump())
