from .utils import OptionDescription, Greeks, HEATMAP_CELL_COUNT
from .pricing_models import (
    compute_black_scholes_prices,
    compute_greeks_black_scholes,
    compute_binomial_prices,
    compute_binomial_greeks,
)

import numpy as np
from scipy.stats import norm
from typing import Dict, Tuple


class Model:
    def __init__(self):
        self.call_greeks: Greeks = None
        self.put_greeks: Greeks = None
        self.call_price = None
        self.put_price = None

        # Avoids recomputing with Black-Scholes when computing the greeks
        self._d1 = None
        self._d2 = None

    def compute_prices(self, option_description: OptionDescription, steps=1000) -> None:
        if option_description.pricing_model == "Black-Scholes":
            prices = compute_black_scholes_prices(option_description=option_description)
            self.call_price = prices["call_price"]
            self.put_price = prices["put_price"]

            # Compute greeks
            self.call_greeks, self.put_greeks = compute_greeks_black_scholes(
                option_description=option_description
            )

        elif option_description.pricing_model == "Cox-Ross-Rubinstein (Binomial)":
            prices = compute_binomial_prices(
                option_description=option_description, steps=steps
            )
            self.call_price = prices["call_price"]
            self.put_price = prices["put_price"]

            # Compute Greeks
            self.call_greeks, self.put_greeks = compute_binomial_greeks(
                option_description=option_description, steps=steps
            )

        else:
            raise ValueError(
                f"Model :: compute_prices :: The pricing model {option_description.pricing_model} is unknown."
            )

    def compute_heatmap_price_grids(
        self,
        option_description: OptionDescription,
        min_spot: float,
        max_spot: float,
        min_vol: float,
        max_vol: float,
    ) -> Dict[str, np.ndarray]:
        spot_range = np.linspace(min_spot, max_spot, HEATMAP_CELL_COUNT)
        volatility_range = np.linspace(min_vol, max_vol, HEATMAP_CELL_COUNT)

        if option_description.pricing_model == "Black-Scholes":
            call_prices, put_prices = self.compute_black_scholes_heatmap_price_grids(
                option_description=option_description,
                spot_range=spot_range,
                volatility_range=volatility_range,
            )
        elif option_description.pricing_model == "Cox-Ross-Rubinstein (Binomial)":
            call_prices, put_prices = self.compute_binomial_heatmap_price_grids(
                option_description=option_description,
                spot_range=spot_range,
                volatility_range=volatility_range,
            )
        else:
            raise ValueError(
                f"Model :: compute_heatmap_price_grids :: The pricing model {option_description.pricing_model} is unknown."
            )
        return {
            "call_prices": call_prices,
            "put_prices": put_prices,
            "spot_range": spot_range,
            "volatility_range": volatility_range,
        }

    def compute_black_scholes_heatmap_price_grids(
        self,
        option_description: OptionDescription,
        spot_range: np.ndarray,
        volatility_range: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        call_prices = np.zeros((len(volatility_range), len(spot_range)))
        put_prices = np.zeros((len(volatility_range), len(spot_range)))

        for i, vol in enumerate(volatility_range):
            for j, spot in enumerate(spot_range):
                option_description_temp = OptionDescription(
                    spot_price=spot,
                    strike_price=option_description.strike_price,
                    risk_free_rate=option_description.risk_free_rate,
                    volatility=vol,
                    maturity=option_description.maturity,
                    pricing_model=option_description.pricing_model,
                    american=option_description.american,
                )
                prices = compute_black_scholes_prices(
                    option_description=option_description_temp
                )
                call_prices[i, j] = prices["call_price"]
                put_prices[i, j] = prices["put_price"]

        return call_prices, put_prices

    def compute_binomial_heatmap_price_grids(
        self,
        option_description: OptionDescription,
        spot_range: np.ndarray,
        volatility_range: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        call_prices = np.zeros((len(volatility_range), len(spot_range)))
        put_prices = np.zeros((len(volatility_range), len(spot_range)))

        for i, vol in enumerate(volatility_range):
            for j, spot in enumerate(spot_range):
                option_description_temp = OptionDescription(
                    spot_price=spot,
                    strike_price=option_description.strike_price,
                    risk_free_rate=option_description.risk_free_rate,
                    volatility=vol,
                    maturity=option_description.maturity,
                    pricing_model=option_description.pricing_model,
                    american=option_description.american,
                )
                prices = compute_binomial_prices(
                    option_description=option_description_temp,
                    steps=100,  # for performance's sake, we keep low.
                )
                call_prices[i, j] = prices["call_price"]
                put_prices[i, j] = prices["put_price"]

        return call_prices, put_prices

    @property
    def get_call_greeks(self) -> Greeks:
        if self.call_greeks is not None:
            return self.call_greeks
        else:
            raise ValueError("Model :: The greeks have not been computed yet !")

    @property
    def get_put_greeks(self) -> Greeks:
        if self.put_greeks is not None:
            return self.put_greeks
        else:
            raise ValueError("Model :: The greeks have not been computed yet !")

    @property
    def get_call_price(self) -> float:
        if self.call_price is not None:
            return self.call_price
        else:
            raise ValueError("Model :: Prices have not been computed yet !")

    @property
    def get_put_price(self) -> float:
        if self.put_price is not None:
            return self.put_price
        else:
            raise ValueError("Model :: Prices have not been computed yet !")
