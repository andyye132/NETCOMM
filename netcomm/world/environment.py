ENV_PARAMS = {
    "open": dict(
        cluster_kappa=0.5, m_0=2.5, alpha_pl=2.2, beta_pl=1e-3,
        env_a=4.88, env_b=0.43,
    ),
    "urban": dict(
        cluster_kappa=1.5, m_0=1.5, alpha_pl=3.5, beta_pl=1e-4,
        env_a=12.0, env_b=0.135,
    ),
    "foliage": dict(
        cluster_kappa=1.0, m_0=1.2, alpha_pl=3.8, beta_pl=5e-5,
        env_a=11.95, env_b=0.136,
    ),
    "indoor": dict(
        cluster_kappa=2.0, m_0=1.0, alpha_pl=4.0, beta_pl=1e-4,
        env_a=27.0, env_b=0.08,
    ),
    "corridor": dict(
        cluster_kappa=1.2, m_0=1.5, alpha_pl=2.8, beta_pl=2e-4,
        env_a=12.0, env_b=0.135,
    ),
}


def get_env_params(env_name: str):
    if env_name not in ENV_PARAMS:
        raise KeyError(f"Unknown env {env_name}; choices: {list(ENV_PARAMS)}")
    return ENV_PARAMS[env_name]
