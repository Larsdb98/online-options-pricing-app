from .utils import Greeks, OptionDescription

import numpy as np
import copy
from typing import Dict, Tuple
from scipy.stats import norm

##################################
### BLACK SCHOLES


def compute_black_scholes_prices(
    option_description: OptionDescription,
) -> Dict[str, float]:
    S = option_description.spot_price
    K = option_description.strike_price
    r = option_description.risk_free_rate
    T = option_description.maturity
    sigma = option_description.volatility

    eps = np.finfo(float).tiny
    sqrtT = np.sqrt(
        np.maximum(T, 0.0)
    )  # TODO: remove this and add validation in OptionDescription pydantic model
    denom = np.maximum(sigma * sqrtT, eps)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / denom

    d2 = d1 - sigma * np.sqrt(T)

    call_price = S * norm.cdf(d1) - (K * np.exp(-(r * T)) * norm.cdf(d2))
    put_price = (K * np.exp(-(r * T)) * norm.cdf(-d2)) - S * norm.cdf(-d1)

    return {"call_price": call_price, "put_price": put_price}


def compute_greeks_black_scholes(
    option_description: OptionDescription,
) -> Tuple[Greeks, Greeks]:
    S = option_description.spot_price
    K = option_description.strike_price
    r = option_description.risk_free_rate
    T = option_description.maturity
    sigma = option_description.volatility

    eps = np.finfo(float).tiny
    sqrtT = np.sqrt(
        np.maximum(T, 0.0)
    )  # TODO: remove this and add validation in OptionDescription pydantic model
    denom = np.maximum(sigma * sqrtT, eps)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / denom

    d2 = d1 - sigma * np.sqrt(T)

    Nd1, Nmd1 = norm.cdf(d1), norm.cdf(-d1)
    Nd2, Nmd2 = norm.cdf(d2), norm.cdf(-d2)
    nd1 = norm.pdf(d1)
    disc = np.exp(-r * T)

    # Base (continuous-time) Greeks for T>0, volatility>0
    gamma = gamma = nd1 / (S * np.maximum(sigma * sqrtT, eps))
    vega = S * nd1 * sqrtT  # per unit volatility

    theta_call = -(S * nd1 * sigma) / (2.0 * sqrtT) - r * K * disc * Nd2
    theta_put = -(S * nd1 * sigma) / (2.0 * sqrtT) + r * K * disc * Nmd2

    rho_call = K * T * disc * Nd2
    rho_put = -K * T * disc * Nmd2

    # Deltas (piecewise at expiry)
    delta_call = Nd1
    delta_put = Nd1 - 1.0

    # At expiry T=0: use limits (intrinsic only; sensitivities mostly 0 except delta)
    at_expiry = T <= 0.0
    if np.any(T <= 0.0):
        itm_call = S > K
        itm_put = S < K

        # Delta at expiry: 1 if ITM for call, 0 otherwise (0.5 at-the-money is undefined; choose 0.5)
        delta_call = np.where(
            at_expiry,
            np.where(itm_call, 1.0, np.where(S == K, 0.5, 0.0)),
            delta_call,
        )
        delta_put = np.where(
            at_expiry,
            np.where(itm_put, -1.0, np.where(S == K, -0.5, 0.0)),
            delta_put,
        )

        # The rest go to ~0 at maturity
        gamma = np.where(at_expiry, 0.0, gamma)
        vega = np.where(at_expiry, 0.0, vega)
        theta_call = np.where(at_expiry, 0.0, theta_call)
        theta_put = np.where(at_expiry, 0.0, theta_put)
        rho_call = np.where(at_expiry, 0.0, rho_call)
        rho_put = np.where(at_expiry, 0.0, rho_put)

    # If volatility is zero, greeks besides delta/rho collapse
    near_zero_vol = np.abs(sigma) < 1e-12
    if np.any(near_zero_vol):
        gamma = np.where(near_zero_vol, 0.0, gamma)
        vega = np.where(near_zero_vol, 0.0, vega)

    call_greeks = Greeks(
        delta=delta_call, gamma=gamma, vega=vega, theta=theta_call, rho=rho_call
    )
    put_greeks = Greeks(
        delta=delta_put, gamma=gamma, vega=vega, theta=theta_put, rho=rho_put
    )
    return call_greeks, put_greeks


##################################
### Cox-Ross-Rubinstein (Binomial)


def compute_binomial_prices(
    option_description: OptionDescription, steps: int = 100
) -> Dict[str, float]:
    """Prices a European or American call and put option using the Cox-Ross-Rubinstein binomial model."""
    S = option_description.spot_price
    K = option_description.strike_price
    r = option_description.risk_free_rate
    T = option_description.maturity
    sigma = option_description.volatility
    american = option_description.american

    dt = T / steps
    if dt <= 0:
        # If no time to maturity, return intrinsic value
        call_price = max(S - K, 0.0)
        put_price = max(K - S, 0.0)
        return {"call_price": call_price, "put_price": put_price}

    # Up and down factors
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u

    # Risk-neutral probability
    p = (np.exp(r * dt) - d) / (u - d)
    if not (0 <= p <= 1):
        raise ValueError(
            "Risk-neutral probability out of range. Adjust parameters or steps."
        )

    # Precompute asset prices at maturity
    ST = np.array([S * (u**j) * (d ** (steps - j)) for j in range(steps + 1)])

    # Option payoffs at maturity
    call_values = np.maximum(ST - K, 0.0)
    put_values = np.maximum(K - ST, 0.0)

    # Step backwards through the tree
    disc = np.exp(-r * dt)
    for i in range(steps - 1, -1, -1):
        ST = ST[: i + 1] / u

        call_values = disc * (p * call_values[1:] + (1 - p) * call_values[:-1])
        put_values = disc * (p * put_values[1:] + (1 - p) * put_values[:-1])

        if american:
            # Early exercise check
            call_values = np.maximum(call_values, ST - K)
            put_values = np.maximum(put_values, K - ST)

    return {"call_price": call_values[0], "put_price": put_values[0]}


# def compute_binomial_greeks_original(
#     option_description: OptionDescription,
#     steps: int = 100,
# ) -> Tuple[Greeks, Greeks]:
#     """There are numerical errors in this method... So rewrote the pipeline using central finite differences."""
#     S = option_description.spot_price
#     K = option_description.strike_price
#     r = option_description.risk_free_rate
#     T = option_description.maturity
#     sigma = option_description.volatility
#     american = option_description.american

#     dt = T / steps
#     u = np.exp(sigma * np.sqrt(dt))
#     d = 1 / u
#     p = (np.exp(r * dt) - d) / (u - d)
#     disc = np.exp(-r * dt)

#     # terminal prices
#     ST = np.array([S * (u**j) * (d ** (steps - j)) for j in range(steps + 1)])

#     values_call = np.maximum(ST - K, 0.0)

#     values_put = np.maximum(K - ST, 0.0)

#     delta_call, gamma_call, theta_call = compute_binomial_delta_gamma_theta(
#         steps=steps, disc=disc, p=p, u=u, dt=dt, values=values_call, ST=ST
#     )
#     delta_put, gamma_put, theta_put = compute_binomial_delta_gamma_theta(
#         steps=steps, disc=disc, p=p, u=u, dt=dt, values=values_put, ST=ST
#     )

#     # Vega (per 1.0 vol point)
#     bump_vol = 0.01

#     V_plus_option_description = OptionDescription(
#         spot_price=S,
#         strike_price=K,
#         risk_free_rate=r,
#         maturity=T,
#         volatility=sigma + bump_vol,
#         american=american,
#     )
#     V_minus_option_description = OptionDescription(
#         spot_price=S,
#         strike_price=K,
#         risk_free_rate=r,
#         maturity=T,
#         volatility=sigma - bump_vol,
#         american=american,
#     )

#     V_plus = compute_binomial_prices(
#         option_description=V_plus_option_description, steps=steps
#     )
#     V_minus = compute_binomial_prices(
#         option_description=V_minus_option_description, steps=steps
#     )

#     vega_call = (V_plus["call_price"] - V_minus["call_price"]) / (2 * bump_vol)
#     vega_put = (V_plus["put_price"] - V_minus["put_price"]) / (2 * bump_vol)

#     # Rho (per 1.0 rate point)
#     bump_r = 0.0001

#     V_plus_r_option_description = OptionDescription(
#         spot_price=S,
#         strike_price=K,
#         risk_free_rate=r + bump_r,
#         maturity=T,
#         volatility=sigma,
#         american=american,
#     )
#     V_minus_r_option_description = OptionDescription(
#         spot_price=S,
#         strike_price=K,
#         risk_free_rate=r - bump_r,
#         maturity=T,
#         volatility=sigma,
#         american=american,
#     )

#     V_plus_r = compute_binomial_prices(
#         option_description=V_plus_r_option_description, steps=steps
#     )
#     V_minus_r = compute_binomial_prices(
#         option_description=V_minus_r_option_description, steps=steps
#     )

#     rho_call = (V_plus_r["call_price"] - V_minus_r["call_price"]) / (2 * bump_r)
#     rho_put = (V_plus_r["put_price"] - V_minus_r["put_price"]) / (2 * bump_r)

#     call_greeks = Greeks(
#         delta=delta_call,
#         gamma=gamma_call,
#         vega=vega_call,
#         theta=theta_call,
#         rho=rho_call,
#     )
#     put_greeks = Greeks(
#         delta=delta_put,
#         gamma=gamma_put,
#         vega=vega_put,
#         theta=theta_put,
#         rho=rho_put,
#     )
#     return call_greeks, put_greeks


def compute_binomial_greeks(
    option_description: OptionDescription,
    steps: int = 100,
    rel_eps_s: float = 1e-4,
    eps_r: float = 1e-4,
    eps_sig: float = 1e-4,
    eps_t_days: int = 1,
) -> Tuple[Greeks, Greeks]:
    """
    Compute Greeks for call and put using central finite differences (bump-and-reprice)

    - Theta returned as per-year sensitivity (same units as price per 1.0 of time).
      If you prefer theta per day, divide by 365.
    """

    # helper to copy and modify the OptionDescription
    def price_with(overrides: dict):
        od = copy.deepcopy(option_description)
        for k, v in overrides.items():
            setattr(od, k, v)
        return compute_binomial_prices(od, steps)

    # baseline
    base = price_with({})  # {"call_price": ..., "put_price": ...}
    S = option_description.spot_price
    sigma = option_description.volatility
    r = option_description.risk_free_rate
    T = option_description.maturity

    # bumps
    dS = max(1e-8, abs(S) * rel_eps_s)  # small absolute bump for spot
    dr = eps_r  # e.g. 1e-4
    dsig = max(1e-8, abs(sigma) * eps_sig)  # small relative bump for vol
    # theta bump: use eps_t_days days expressed in years (assuming T in years)
    dT = max(1e-12, eps_t_days / 365.0)

    # --- Delta & Gamma ---
    up = price_with({"spot_price": S + dS})
    down = price_with({"spot_price": S - dS})

    call_up = up["call_price"]
    call_down = down["call_price"]
    put_up = up["put_price"]
    put_down = down["put_price"]

    # central finite difference for delta
    delta_call = (call_up - call_down) / (2 * dS)
    delta_put = (put_up - put_down) / (2 * dS)

    # second derivative for gamma
    gamma_call = (call_up - 2.0 * base["call_price"] + call_down) / (dS * dS)
    gamma_put = (put_up - 2.0 * base["put_price"] + put_down) / (dS * dS)

    # --- Theta (time decay) ---
    # price at slightly earlier maturity (T - dT). If T-dT <= 0, compute_binomial_prices handles intrinsic.
    price_T_minus = price_with({"maturity": max(0.0, T - dT)})
    # Theta ≈ (V(T - dT) - V(T)) / dT (per year)
    theta_call = (price_T_minus["call_price"] - base["call_price"]) / dT
    theta_put = (price_T_minus["put_price"] - base["put_price"]) / dT

    # --- Vega (volatility sensitivity) ---
    price_sig_up = price_with({"volatility": sigma + dsig})
    price_sig_down = price_with({"volatility": max(0.0, sigma - dsig)})
    vega_call = (price_sig_up["call_price"] - price_sig_down["call_price"]) / (2 * dsig)
    vega_put = (price_sig_up["put_price"] - price_sig_down["put_price"]) / (2 * dsig)

    # --- Rho (rate sensitivity) ---
    price_r_up = price_with({"risk_free_rate": r + dr})
    price_r_down = price_with({"risk_free_rate": r - dr})
    rho_call = (price_r_up["call_price"] - price_r_down["call_price"]) / (2 * dr)
    rho_put = (price_r_up["put_price"] - price_r_down["put_price"]) / (2 * dr)

    call_greeks = Greeks(
        delta=delta_call,
        gamma=gamma_call,
        vega=vega_call,
        theta=theta_call,
        rho=rho_call,
    )
    put_greeks = Greeks(
        delta=delta_put, gamma=gamma_put, vega=vega_put, theta=theta_put, rho=rho_put
    )

    return call_greeks, put_greeks


# helper
def compute_binomial_delta_gamma_theta(
    steps, disc, p, u, dt, values, ST
) -> Tuple[float, float, float]:
    for i in range(steps - 1, -1, -1):
        ST = ST[: i + 1] / u
        values = disc * (p * values[1:] + (1 - p) * values[:-1])

        if i == 2:
            Vuu, Vu, Vd = values[0], values[1], values[2]
            Su, Sm, Sd = ST[0], ST[1], ST[2]
        if i == 1:
            Vu, Vd = values[0], values[1]
            Su, Sd = ST[0], ST[1]
        if i == 0:
            V0 = values[0]

    delta = (Vu - Vd) / (Su - Sd)

    delta_u = (Vuu - Vu) / (Su - Sm)
    delta_d = (Vu - Vd) / (Sm - Sd)
    gamma = (delta_u - delta_d) / ((Su - Sd) / 2)

    # theta (per year)
    theta = (Vuu - V0) / (2 * dt)  # simplified binomial theta approximation

    return delta, gamma, theta
