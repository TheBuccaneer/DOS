"""Analytical examples for H7, with invented parameters explicitly labeled.

No generated latency is represented as a serving measurement. No RNG, server,
GPU, network, trained classifier or empirical attack success rate is used.
"""
import json
import math
from pathlib import Path


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def gaussian(delta, sigma, reference_noise, private_noise, correlated=True):
    """T=delta*B+W+N; Z=W+E, all independent zero-mean Gaussian terms.

    B has equal priors. For the negative control Z is independent of W.
    Report exact optimal balanced classification accuracy under this model.
    """
    marginal_var = sigma**2 + private_noise**2
    if not correlated or math.isinf(reference_noise):
        conditional_var = marginal_var
    else:
        conditional_var = (private_noise**2 +
                           sigma**2 * reference_noise**2 / (sigma**2 + reference_noise**2))
    marginal_acc = phi(delta / (2 * math.sqrt(marginal_var)))
    conditional_acc = phi(delta / (2 * math.sqrt(conditional_var)))
    return dict(marginal_overlap=2 * (1 - marginal_acc),
                marginal_balanced_bayes_accuracy=marginal_acc,
                conditional_balanced_bayes_accuracy=conditional_acc)


def uniform(delta, width, reference_error_bound):
    """T=delta*B+W; W uniform [0,width], Z=W+E, |E|<=bound.

    Residual supports [-e,e] and [delta-e,delta+e] are disjoint if delta>2e.
    No independence assumption about E is needed for this support argument.
    """
    overlap = max(0, 1 - delta / width)
    return dict(marginal_overlap=overlap,
                marginal_balanced_bayes_accuracy=1 - overlap / 2,
                residual_supports=[[-reference_error_bound, reference_error_bound],
                                   [delta-reference_error_bound, delta+reference_error_bound]],
                residual_supports_disjoint=delta > 2 * reference_error_bound)


if __name__ == '__main__':
    out = dict(status='analytical counterexample only; parameters are constructed, not fitted',
               units='arbitrary milliseconds for illustration',
               uniform_parameters=dict(delta=20, width=1000, reference_error_bound=2),
               uniform_result=uniform(20, 1000, 2),
               gaussian_parameters=dict(delta=20, sigma=200, reference_noise=2, private_noise=2),
               gaussian_correlated=gaussian(20, 200, 2, 2),
               gaussian_independent_reference=gaussian(20, 200, 2, 2, correlated=False),
               gaussian_reference_uninformative=gaussian(20, 200, math.inf, 2),
               inference='marginal overlap alone cannot bound inference with a correlated reference',
               not_established=['availability of the assumed reference through an LLM API',
                                'a live defense bypass', 'novelty beyond existing timing literature'])
    root = Path(__file__).resolve().parents[1]
    (root / 'evidence/conditional_observer.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))
