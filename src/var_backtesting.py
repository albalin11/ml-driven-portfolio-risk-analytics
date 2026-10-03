"""Small shared calculation used by Part 4 and its tests."""
import numpy as np
from scipy.special import xlogy
from scipy.stats import chi2


def kupiec_test(violations, expected_rate):
    """Return Kupiec's unconditional-coverage statistic and p-value."""
    flags = np.asarray(violations, dtype=bool)
    observations = flags.size
    if observations == 0:
        raise ValueError('Kupiec test requires at least one observation.')
    if not 0 < expected_rate < 1:
        raise ValueError('Expected violation rate must be between zero and one.')

    count = int(flags.sum())
    observed_rate = count / observations
    null_log_likelihood = (
        xlogy(count, expected_rate)
        + xlogy(observations - count, 1 - expected_rate)
    )
    fitted_log_likelihood = (
        xlogy(count, observed_rate)
        + xlogy(observations - count, 1 - observed_rate)
    )
    statistic = max(0.0, float(2 * (fitted_log_likelihood - null_log_likelihood)))
    p_value = float(chi2.sf(statistic, df=1))
    return statistic, p_value
