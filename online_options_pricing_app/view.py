from .model import Model
from .utils import PRICING_MODELS, Greeks

import panel as pn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


class View:
    def __init__(
        self,
        model: Model,
        css_style_path: Path = None,
        dashboard_title: str = "Online Option Pricing Application",
    ):
        self.model = model
        self.dashboard_title = dashboard_title
        self.title_widget = pn.pane.Markdown(
            "# Online Option Pricing Application", css_classes=["app-title"]
        )
        self.subtitle_widget = pn.pane.Markdown(
            "Compute call/put option prices using Black-Scholes or the Binomial pricing models.",
            css_classes=["app-sub"],
        )
        if css_style_path is not None:
            with open(css_style_path, "r") as f:
                self.css = f.read()
        else:
            self.css = ""

        pn.extension(
            "plotly", "mathjax", raw_css=[self.css], sizing_mode="stretch_width"
        )

        ##### INPUTS
        self.input_labels_dict = {
            "spot": pn.pane.Markdown("**Spot Price**", css_classes=["input-text"]),
            "strike": pn.pane.Markdown("**Strike Price**", css_classes=["input-text"]),
            "risk": pn.pane.Markdown("**Risk-Free Rate**", css_classes=["input-text"]),
            "volatility": pn.pane.Markdown(
                "**Volatility (σ)**", css_classes=["input-text"]
            ),
            "maturity": pn.pane.Markdown(
                "**Time to Maturity (years)**", css_classes=["input-text"]
            ),
            "pricing_model": pn.pane.Markdown(
                "**Pricing Model**", css_classes=["input-text"]
            ),
            "heatmap_settings_subtitle": pn.pane.Markdown(
                "## Heatmap Settings", css_classes=["input-text"]
            ),
            "min_spot_price_heatmap": pn.pane.Markdown(
                "**Minimum Spot Price**", css_classes=["input-text"]
            ),
            "max_spot_price_heatmap": pn.pane.Markdown(
                "**Maximum Spot Price**", css_classes=["input-text"]
            ),
            "min_volatility_heatmap": pn.pane.Markdown(
                "**Minimum Volatility for Heatmaps**", css_classes=["input-text"]
            ),
            "max_volatility_heatmap": pn.pane.Markdown(
                "**Maximum Volatility for Heatmaps**", css_classes=["input-text"]
            ),
        }

        self.spot_price_input = pn.widgets.FloatInput(
            value=100.0, step=0.1, format="10.4f"
        )
        self.strike_price_input = pn.widgets.FloatInput(
            value=105.0, step=0.1, format="10.4f"
        )
        self.risk_free_input = pn.widgets.FloatInput(
            value=0.05,
            step=0.01,
            format="1.4f",
        )
        self.sigma_input = pn.widgets.FloatInput(
            value=0.2,
            step=0.01,
            format="1.4f",
        )
        self.time_to_maturity_input = pn.widgets.FloatInput(
            value=0.5,
            step=0.1,
            format="1.2f",
        )

        self.pricing_model_input = pn.widgets.Select(options=PRICING_MODELS)

        self.option_is_american_input = pn.widgets.Toggle(
            name="American Option",
            button_style="outline",
            button_type="success",
            value=False,
            css_classes=["custom-toggle"],
        )

        self.min_spot_price_input = pn.widgets.FloatInput(
            value=80.0, step=0.1, format="10.2f"
        )
        self.max_spot_price_input = pn.widgets.FloatInput(
            value=120.0, step=0.1, format="10.2f"
        )
        self.min_volatility_slider = pn.widgets.FloatSlider(
            start=0.0, end=1.0, step=0.01, value=0.2, css_classes=["input-text"]
        )
        self.max_volatility_slider = pn.widgets.FloatSlider(
            start=0.0, end=1.0, step=0.01, value=0.8, css_classes=["input-text"]
        )

        ##### OUTPUTS
        self.price_output = pn.pane.Markdown(
            "**Option Price:** -", css_classes=["price-text"]
        )
        self.greeks_output = pn.pane.Markdown(
            "**Greeks:** -", css_classes=["greeks-text"]
        )

        self.call_price_card = pn.pane.Markdown(
            "### Call Price\n\n$0.0",
            css_classes=["call-price-card"],
            sizing_mode="stretch_width",
        )

        self.put_price_card = pn.pane.Markdown(
            "### Put Price\n\n$0.0",
            css_classes=["put-price-card"],
            sizing_mode="stretch_width",
        )

        # temporary
        greeks = Greeks(
            delta=0.0, gamma=0.0, vega=0.0, theta=-0.0, rho=0.0
        )  # Inital dummy load.

        self.call_greeks_table = pn.pane.Markdown(
            self.generate_greeks_table(greeks=greeks, greek_type="Call"),
            css_classes=["greeks-table"],
            sizing_mode="stretch_width",
        )

        self.put_greeks_table = pn.pane.Markdown(
            self.generate_greeks_table(greeks=greeks, greek_type="Put"),
            css_classes=["greeks-table"],
            sizing_mode="stretch_width",
        )

        ##### LAYOUT

        self.price_row = pn.Row(
            self.call_price_card, self.put_price_card, sizing_mode="stretch_width"
        )

        # Dummy plots until it works
        heatmap_fig = self.generate_option_heatmaps(
            spot_range=np.array([0.0, 0.1, 0.2]),
            volatility_range=np.array([0.0, 0.1, 0.2]),
            call_prices=np.array([0.0, 0.0, 0.0]),
            put_prices=np.array([0.0, 0.0, 0.0]),
        )
        self.heatmap_pane = pn.pane.Plotly(
            heatmap_fig, config={"responsive": True}, sizing_mode="stretch_width"
        )

        self.left_col = pn.Column(
            self.title_widget,
            self.subtitle_widget,
            pn.Spacer(height=6),
            self.input_labels_dict["spot"],
            self.spot_price_input,
            self.input_labels_dict["strike"],
            self.strike_price_input,
            self.input_labels_dict["risk"],
            self.risk_free_input,
            self.input_labels_dict["volatility"],
            self.sigma_input,
            self.input_labels_dict["maturity"],
            self.time_to_maturity_input,
            self.input_labels_dict["pricing_model"],
            self.pricing_model_input,
            self.option_is_american_input,
            pn.Spacer(height=6),
            pn.layout.Divider(),
            self.input_labels_dict["heatmap_settings_subtitle"],
            self.input_labels_dict["min_spot_price_heatmap"],
            self.min_spot_price_input,
            self.input_labels_dict["max_spot_price_heatmap"],
            self.max_spot_price_input,
            self.input_labels_dict["min_volatility_heatmap"],
            self.min_volatility_slider,
            self.input_labels_dict["max_volatility_heatmap"],
            self.max_volatility_slider,
            css_classes=["side-panel"],
            width=360,
        )

        self.right_col = pn.Column(
            pn.pane.Markdown("## Results", css_classes=["app-title"]),
            self.price_row,
            self.call_greeks_table,
            self.put_greeks_table,
            pn.Spacer(height=8),
            pn.pane.Markdown("## Call/Put Price Heatmaps", css_classes=["app-sub"]),
            self.heatmap_pane,
            css_classes=["results-panel"],
            sizing_mode="stretch_both",
        )

        self.dashboard = pn.Row(
            self.left_col, self.right_col, sizing_mode="stretch_both"
        )

    def generate_option_heatmaps(
        self,
        spot_range: np.ndarray,
        volatility_range: np.ndarray,
        call_prices: np.ndarray,
        put_prices: np.ndarray,
    ):
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Call Prices", "Put Prices"),
            horizontal_spacing=0.25,
        )

        fig.add_trace(
            go.Heatmap(
                z=call_prices,
                x=spot_range,
                y=volatility_range,
                colorscale="Viridis",
                colorbar=dict(title="Price", x=0.46),
                text=call_prices,
                texttemplate="$%{z:.2f}",
                textfont={"color": "black", "size": 9},
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Heatmap(
                z=put_prices,
                x=spot_range,
                y=volatility_range,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Price", x=1.05),
                text=put_prices,
                texttemplate="$%{z:.2f}",
                textfont={"color": "black", "size": 9},
            ),
            row=1,
            col=2,
        )

        fig.update_xaxes(title_text="Spot Price", row=1, col=1)
        fig.update_xaxes(title_text="Spot Price", row=1, col=2)
        fig.update_yaxes(title_text="Volatility", row=1, col=1)
        fig.update_yaxes(title_text="Volatility", row=1, col=2)

        fig.update_layout(
            # title_text="Option Pricing Heatmaps (Dummy Data)",
            template="plotly_dark",
            height=600,
        )
        return fig

    def update_greeks_tables(self, call_greeks: Greeks, put_greeks: Greeks) -> None:
        """
        Update the greeks table with new values.
        """
        table_md_call = self.generate_greeks_table(
            greeks=call_greeks, greek_type="Call"
        )
        table_md_put = self.generate_greeks_table(greeks=put_greeks, greek_type="Put")

        self.call_greeks_table.object = table_md_call
        self.put_greeks_table.object = table_md_put

    def generate_greeks_table(self, greeks: Greeks, greek_type: str = "Call") -> str:
        """
        Create a Markdown table for Greeks with LaTeX headers.
        """
        headers = r"$$\Delta$$ | $$\Gamma$$ | $$\Theta$$ | $$\nu$$ | $$\rho$$"
        sep = " | ".join(["---"] * 5)  # we only display the 5 greeks above
        # values = " | ".join([f"{v:.4f}" for v in greeks.values()])
        values = f"{greeks.delta:.4f} | {greeks.gamma:.4f} | {greeks.theta:.4f} | {greeks.vega:.4f} | {greeks.rho:.4f}"

        table_md = f"""
### {greek_type} Greeks
<style>
table {{
  width: 100%;
  table-layout: fixed;
}}
th, td {{
  text-align: center;
  padding: 6px;
}}
</style>

| {headers} |
| {sep} |
| {values} |
"""

        return table_md

    def run(self) -> None:
        pn.serve(self.dashboard, title=self.dashboard_title, show=True)
