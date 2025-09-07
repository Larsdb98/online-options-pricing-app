from pydantic import BaseModel

PRICING_MODELS = ["Black-Scholes", "Cox-Ross-Rubinstein (Binomial)"]
HEATMAP_CELL_COUNT = 10


class OptionDescription(BaseModel):
    spot_price: float
    strike_price: float
    risk_free_rate: float
    volatility: float
    maturity: float
    pricing_model: str = "Black-Scholes"
    american: bool = False


class Greeks(BaseModel):
    delta: float = None
    gamma: float = None
    vega: float = None
    theta: float = None
    rho: float = None
