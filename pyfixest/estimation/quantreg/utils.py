from scipy.stats import norm


def get_hall_sheather_bandwidth(q: float, N: int, alpha: float = 0.05) -> float:
    "Compute the Hall-Sheather bandwidth."
    x0 = norm.ppf(q)
    f0 = norm.pdf(x0)

    h = (
        N ** (-1 / 3)
        * norm.ppf(1 - alpha / 2) ** (2 / 3)
        * ((1.5 * f0**2) / (2 * x0**2 + 1)) ** (1 / 3)
    )

    return h
