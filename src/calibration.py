from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def ece_equal_width(labels, probabilities, bins=15):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    confidence = probabilities.max(1)
    correct = (probabilities.argmax(1) == labels).astype(float)
    edges, value = np.linspace(0, 1, bins + 1), 0.0
    for index in range(bins):
        mask = ((confidence >= edges[index]) if index == 0 else (confidence > edges[index])) & (confidence <= edges[index + 1])
        if mask.any():
            value += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    return float(value)


def ece_equal_mass(labels, probabilities, bins=15):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    confidence = probabilities.max(1)
    correct = (probabilities.argmax(1) == labels).astype(float)
    order = np.argsort(confidence)
    return float(sum(len(ids) / len(order) * abs(confidence[ids].mean() - correct[ids].mean()) for ids in np.array_split(order, min(bins, len(order))) if len(ids)))


ESTIMATORS = {"EW15": lambda y, p: ece_equal_width(y, p, 15), "EM15": lambda y, p: ece_equal_mass(y, p, 15), "EW30": lambda y, p: ece_equal_width(y, p, 30)}


def fit_temperature(logits, labels):
    logits, labels = np.asarray(logits), np.asarray(labels)
    def objective(log_temperature):
        scaled = logits / np.exp(log_temperature)
        scaled -= scaled.max(1, keepdims=True)
        return float(-np.mean(scaled[np.arange(len(labels)), labels] - np.log(np.exp(scaled).sum(1))))
    return float(np.exp(minimize_scalar(objective, bounds=(-3, 3), method="bounded").x))


def aggregate_metrics(eces, weights, tail_alpha=0.20):
    eces, weights = np.asarray(eces, float), np.asarray(weights, float)
    valid = np.isfinite(eces)
    eces, weights = eces[valid], weights[valid]
    weights /= weights.sum()
    cwece = float(weights @ eces)
    tail_count = max(1, int(np.ceil(tail_alpha * len(eces))))
    tail = float(np.sort(eces)[-tail_count:].mean())
    return {"cwece": cwece, "tailece": tail, "tailgap": tail - cwece, "tailtrap": tail / max(cwece, 1e-12), "worst_ece": float(eces.max())}


def aggregate_metrics_full(eces, weights, tail_alpha=0.20):
    eces, weights = np.asarray(eces, float), np.asarray(weights, float)
    valid = np.isfinite(eces)
    eces, weights = eces[valid], weights[valid]
    weights = weights / weights.sum()
    uniform = float(eces.mean())
    main = aggregate_metrics(eces, weights, tail_alpha)
    covariance = float(np.mean((weights - 1.0 / len(eces)) * (eces - uniform)))
    return {**main, "uwece": uniform, "a_tail_heterogeneity": main["tailece"] / max(uniform, 1e-12), "m_sample_weight_masking": uniform / max(main["cwece"], 1e-12), "k_cov_u_w_ece": len(eces) * covariance, "pearson_r_weight_ece": float(np.corrcoef(weights, eces)[0, 1]) if np.std(weights) and np.std(eces) else np.nan}


def proposition2_stats(eces, weights):
    eces, weights = np.asarray(eces, float), np.asarray(weights, float)
    weights /= weights.sum()
    index = int(eces.argmax())
    bound = (1 - weights[index]) * (eces.max() - eces.min())
    return {"ece_max": float(eces.max()), "ece_min": float(eces.min()), "worst_weight": float(weights[index]), "worst_gap": float(eces.max() - weights @ eces), "prop2_bound": float(bound), "prop2_tightness": float((eces.max() - weights @ eces) / bound) if bound else np.nan}
