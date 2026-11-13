import logging
from pathlib import Path
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Path as ApiPath, Query, Response
from pydantic import BaseModel

from . import config, metrics
from .history_store import HistoryStore, HistoryStoreError
from .inference import FraudScorer
from .redis_store import RedisStore

logger = logging.getLogger(__name__)
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
    state["history_store"] = HistoryStore(config.HISTORY_DB)
    state["history_store_ready"] = False
    try:
        state["history_store"].initialize()
        state["history_store_ready"] = True
    except HistoryStoreError as exc:
        logger.error("Transaction history initialization failed (%s)", type(exc).__name__)
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
STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(STATIC_DIR / "index.html")

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
    history_store = state.get("history_store")
    history_store_ok = bool(history_store and history_store.is_healthy())
    return {
        "status": "ok" if state.get("scorer") is not None else "degraded",
        "redis": redis_ok,
        "model_ready": state.get("scorer") is not None,
        "startup_error": state.get("startup_error"),
        "history_store": history_store_ok,
    }


@app.get("/metrics")
def prometheus_metrics():
    body, content_type = metrics.render()
    return Response(content=body, media_type=content_type)


@app.post("/score")
def score(txn: Transaction):
    payload = txn.model_dump()
    result = score_transaction(payload)
    history_store: HistoryStore | None = state.get("history_store")
    if history_store is not None:
        safe_payload = {
            "TransactionAmt": txn.TransactionAmt,
            "ProductCD": txn.ProductCD,
            "TransactionDT": txn.TransactionDT,
        }
        try:
            result["history_id"] = history_store.save_analysis(safe_payload, result)
        except HistoryStoreError as exc:
            logger.error("Transaction history write failed (%s)", type(exc).__name__)
    return result


@app.get("/history")
def transaction_history(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    search: str | None = Query(default=None, max_length=128),
    decision: Literal["APPROVE", "FLAG", "DECLINE"] | None = None,
):
    history_store: HistoryStore | None = state.get("history_store")
    if history_store is None:
        raise HTTPException(status_code=503, detail="Transaction history is unavailable.")
    try:
        page = history_store.list_records(
            limit=limit,
            offset=offset,
            search=search.strip() if search else None,
            decision=decision,
        )
    except HistoryStoreError as exc:
        logger.error("Transaction history query failed (%s)", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Transaction history is temporarily unavailable.") from None
    return {**page, "limit": limit, "offset": offset}


@app.get("/history/{record_id}")
def transaction_history_detail(record_id: int = ApiPath(ge=1)):
    history_store: HistoryStore | None = state.get("history_store")
    if history_store is None:
        raise HTTPException(status_code=503, detail="Transaction history is unavailable.")
    try:
        record = history_store.get_record(record_id)
    except HistoryStoreError as exc:
        logger.error("Transaction history detail query failed (%s)", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Transaction history is temporarily unavailable.") from None
    if record is None:
        raise HTTPException(status_code=404, detail="Transaction history record not found.")
    return record
