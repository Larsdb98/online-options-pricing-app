from .model import Model
from .view import View
from .utils import OptionDescription

from param.parameterized import Event


class Controller:
    def __init__(self, model: Model, view: View):
        self.model = model
        self.view = view

        self._bind()

        # Initial values pulled from UI
        self.option_description = OptionDescription(
            spot_price=self.view.spot_price_input.value,
            strike_price=self.view.strike_price_input.value,
            risk_free_rate=self.view.risk_free_input.value,
            volatility=self.view.sigma_input.value,
            maturity=self.view.time_to_maturity_input.value,
            pricing_model=self.view.pricing_model_input.value,
            american=self.view.option_is_american_input.value,
        )

        # Compute for initial values
        self.compute_prices_and_greeks()
        self.update_heatmaps()

    def _bind(self):
        # General options inputs
        self.view.spot_price_input.param.watch(
            self.option_description_changed_callback, "value"
        )
        self.view.strike_price_input.param.watch(
            self.option_description_changed_callback, "value"
        )
        self.view.risk_free_input.param.watch(
            self.option_description_changed_callback, "value"
        )
        self.view.sigma_input.param.watch(
            self.option_description_changed_callback, "value"
        )
        self.view.time_to_maturity_input.param.watch(
            self.option_description_changed_callback, "value"
        )
        self.view.pricing_model_input.param.watch(
            self.option_description_changed_callback, "value"
        )
        self.view.option_is_american_input.param.watch(
            self.option_description_changed_callback, "value"
        )

        # heatmap params inputs
        self.view.min_spot_price_input.param.watch(
            self.heatmap_update_callback, "value"
        )
        self.view.max_spot_price_input.param.watch(
            self.heatmap_update_callback, "value"
        )
        self.view.min_volatility_slider.param.watch(
            self.heatmap_update_callback, "value"
        )
        self.view.max_volatility_slider.param.watch(
            self.heatmap_update_callback, "value"
        )

    def option_description_changed_callback(self, event: Event) -> None:
        self.option_description = OptionDescription(
            spot_price=self.view.spot_price_input.value,
            strike_price=self.view.strike_price_input.value,
            risk_free_rate=self.view.risk_free_input.value,
            volatility=self.view.sigma_input.value,
            maturity=self.view.time_to_maturity_input.value,
            pricing_model=self.view.pricing_model_input.value,
            american=self.view.option_is_american_input.value,
        )
        self.compute_prices_and_greeks()
        self.update_heatmaps()

    def heatmap_update_callback(self, event: Event) -> None:
        self.update_heatmaps()

    def compute_prices_and_greeks(self) -> None:
        self.model.compute_prices(option_description=self.option_description, steps=100)
        self.update_prices_and_greeks()

    def update_prices_and_greeks(self) -> None:
        call_price = self.model.get_call_price
        put_price = self.model.get_put_price

        call_greeks = self.model.get_call_greeks
        put_greeks = self.model.get_put_greeks

        self.view.call_price_card.object = f"### Call Price\n\n${call_price:.2f}"
        self.view.put_price_card.object = f"### Put Price\n\n${put_price:.2f}"

        self.view.update_greeks_tables(call_greeks=call_greeks, put_greeks=put_greeks)

    def update_heatmaps(self) -> None:
        min_spot = self.view.min_spot_price_input.value
        max_spot = self.view.max_spot_price_input.value
        min_volatility = self.view.min_volatility_slider.value
        max_volatility = self.view.max_volatility_slider.value

        heatmap_values_dict = self.model.compute_heatmap_price_grids(
            option_description=self.option_description,
            min_spot=min_spot,
            max_spot=max_spot,
            min_vol=min_volatility,
            max_vol=max_volatility,
        )
        plotly_figure = self.view.generate_option_heatmaps(
            spot_range=heatmap_values_dict["spot_range"],
            volatility_range=heatmap_values_dict["volatility_range"],
            call_prices=heatmap_values_dict["call_prices"],
            put_prices=heatmap_values_dict["put_prices"],
        )

        self.view.heatmap_pane.object = plotly_figure
