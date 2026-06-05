from typing import Dict, Optional
import jax.numpy as jnp


def collect_observations(world, last_outcomes: Optional[jnp.ndarray],
                         beacons: Optional[Dict], channel_forecast) -> Dict:
    # why: bundle (N, N) per-link observation channels into a dict the
    # emission likelihood can consume. The runner is responsible for passing
    # whatever it has; missing fields default to neutral values.
    sinr = channel_forecast.mean_sinr  # (N, N, L)
    pi_up_proxy = jnp.clip(jnp.mean(sinr / (sinr + 1.0), axis=-1), 0.0, 1.0)  # (N, N)
    doppler = jnp.mean(channel_forecast.doppler_spread, axis=-1)  # (N, N)
    coh = jnp.mean(channel_forecast.coh_time, axis=-1)  # (N, N)
    N = sinr.shape[0]
    if last_outcomes is None:
        ack = jnp.ones((N, N), dtype=jnp.float32) * pi_up_proxy
    else:
        ack = last_outcomes.astype(jnp.float32)
    rssi = jnp.log1p(jnp.mean(sinr, axis=-1))  # log SINR as RSSI proxy
    delay = 1.0 / (jnp.mean(sinr, axis=-1) + 1e-3)
    loss = 1.0 - pi_up_proxy
    if beacons is None:
        missed = jnp.zeros((N, N), dtype=jnp.float32)
    else:
        missed = beacons.get("missed", jnp.zeros((N, N), dtype=jnp.float32))
    return {
        "rssi": rssi,
        "loss": loss,
        "ack": ack,
        "delay": delay,
        "missed_beacons": missed,
        "rel_vel": jnp.zeros((N, N), dtype=jnp.float32),
        "doppler_est": doppler,
        "coh_time": coh,
    }


def _gauss(x, mu, sigma):
    z = (x - mu) / jnp.clip(sigma, 1e-6, None)
    return jnp.exp(-0.5 * z * z)


def observation_likelihood(obs: Dict, regime_params: Optional[Dict] = None) -> jnp.ndarray:
    # why: per-regime emission product over (RSSI, delay, doppler, ack, missed).
    # Means/stds are hand-picked to encode the qualitative description of each regime
    # from the paper (stable/predictable/volatile/blocked).
    rssi = obs["rssi"]
    delay = obs["delay"]
    doppler = obs["doppler_est"]
    ack = obs["ack"]
    missed = obs["missed_beacons"]

    if regime_params is None:
        regime_params = _default_regime_params()

    likes = []
    for r in ("stable", "predictable", "volatile", "blocked"):
        p = regime_params[r]
        lr = _gauss(rssi, p["rssi_mu"], p["rssi_sigma"])
        ld = _gauss(delay, p["delay_mu"], p["delay_sigma"])
        ldo = _gauss(doppler, p["doppler_mu"], p["doppler_sigma"])
        # Bernoulli on ack
        la = ack * p["ack_p"] + (1.0 - ack) * (1.0 - p["ack_p"])
        # Poisson-rate-ish on missed beacons (treat as exp(-lambda) likelihood)
        lm = jnp.exp(-jnp.abs(missed - p["missed_mu"]))
        likes.append(lr * ld * ldo * la * lm)
    out = jnp.stack(likes, axis=-1)  # (N, N, 4)
    out = out / jnp.clip(jnp.sum(out, axis=-1, keepdims=True), 1e-9, None)
    return out


def _default_regime_params() -> Dict:
    return {
        "stable":      {"rssi_mu": 2.5, "rssi_sigma": 1.0,
                        "delay_mu": 0.1, "delay_sigma": 0.3,
                        "doppler_mu": 5.0, "doppler_sigma": 10.0,
                        "ack_p": 0.95, "missed_mu": 0.0},
        "predictable": {"rssi_mu": 2.0, "rssi_sigma": 1.2,
                        "delay_mu": 0.3, "delay_sigma": 0.4,
                        "doppler_mu": 30.0, "doppler_sigma": 20.0,
                        "ack_p": 0.80, "missed_mu": 0.5},
        "volatile":    {"rssi_mu": 1.0, "rssi_sigma": 1.5,
                        "delay_mu": 0.8, "delay_sigma": 0.6,
                        "doppler_mu": 100.0, "doppler_sigma": 50.0,
                        "ack_p": 0.50, "missed_mu": 2.0},
        "blocked":     {"rssi_mu": 0.1, "rssi_sigma": 0.5,
                        "delay_mu": 2.0, "delay_sigma": 1.0,
                        "doppler_mu": 50.0, "doppler_sigma": 80.0,
                        "ack_p": 0.05, "missed_mu": 5.0},
    }
