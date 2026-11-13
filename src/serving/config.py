import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _project_path(value: str, default: Path) -> Path:
    path = Path(value).expanduser() if value else default
    return path if path.is_absolute() else REPO_ROOT / path


ARTIFACTS = _project_path(os.getenv("FRAUDGRAPH_ARTIFACTS", ""), REPO_ROOT / "artifacts")
HISTORY_DB = _project_path(
    os.getenv("FRAUDGRAPH_HISTORY_DB", ""),
    REPO_ROOT / "data" / "fraudgraph_history.db",
)
DATABASE_URL = os.getenv("DATABASE_URL") or None
PGSSLMODE = os.getenv("PGSSLMODE", "require")
REDIS_URL = os.getenv("REDIS_URL") or None
REDIS_HOST = os.getenv("REDIS_HOST") or None
REDIS_CONFIGURED = bool(REDIS_URL or REDIS_HOST)

MODEL_WEIGHTS = ARTIFACTS / "graphsage_pure.pt"
MODEL_META = ARTIFACTS / "model_meta.pkl"
GRAPH_META = ARTIFACTS / "graph_meta.pkl"
EXPLAINER_REF = ARTIFACTS / "explainer_reference.pkl"

# Decision thresholds (fixed for the project; production would tune to cost).
APPROVE_BELOW = 0.30
DECLINE_ABOVE = 0.70


def decision_for(score: float) -> str:
    if score < APPROVE_BELOW:
        return "APPROVE"
    if score > DECLINE_ABOVE:
        return "DECLINE"
    return "FLAG"


# Kafka
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC_IN = "transactions"
TOPIC_OUT = "fraud-scores"
NUM_PARTITIONS = 3
CONSUMER_GROUP = "fraud-scorer"

# Redis
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
VELOCITY_TTL_SEC = 48 * 3600
NEIGHBOR_LIST_LEN = 20

# Velocity windows (seconds)
WINDOWS = {"count_1h": 3600, "count_6h": 21600, "count_24h": 86400}
