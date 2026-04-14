import numpy as np


def probability_to_strength(probabilities):
    """
    Convert class probabilities into a readable strength level.
    """
    if probabilities is None or len(probabilities) == 0:
        return "Weak"

    top_prob = float(np.max(probabilities))

    if top_prob >= 0.75:
        return "Strong"
    if top_prob >= 0.55:
        return "Moderate"
    return "Weak"


def probability_to_score(probabilities):
    """
    Convert class probabilities into a 0-100 score.
    """
    if probabilities is None or len(probabilities) == 0:
        return 0.0

    top_prob = float(np.max(probabilities))
    return round(top_prob * 100.0, 1)


def normalize_three_class_direction(probabilities):
    """
    Map 3-class direction probabilities into labels.
    Expected order: [DOWN, NEUTRAL, UP]
    """
    if probabilities is None or len(probabilities) != 3:
        return {
            "label": "NEUTRAL",
            "score": 0.0,
            "strength": "Weak",
            "probabilities": {
                "DOWN": 0.0,
                "NEUTRAL": 0.0,
                "UP": 0.0,
            },
        }

    down_prob, neutral_prob, up_prob = [float(x) for x in probabilities]
    top_index = int(np.argmax(probabilities))

    label_map = {
        0: "DOWN",
        1: "NEUTRAL",
        2: "UP",
    }

    label = label_map[top_index]

    return {
        "label": label,
        "score": probability_to_score(probabilities),
        "strength": probability_to_strength(probabilities),
        "probabilities": {
            "DOWN": round(down_prob, 4),
            "NEUTRAL": round(neutral_prob, 4),
            "UP": round(up_prob, 4),
        },
    }


def normalize_three_class_quality(probabilities):
    """
    Map 3-class setup quality probabilities into labels.
    Expected order: [WEAK, MIXED, CLEAN]
    """
    if probabilities is None or len(probabilities) != 3:
        return {
            "label": "MIXED",
            "score": 0.0,
            "strength": "Weak",
            "probabilities": {
                "WEAK": 0.0,
                "MIXED": 0.0,
                "CLEAN": 0.0,
            },
        }

    weak_prob, mixed_prob, clean_prob = [float(x) for x in probabilities]
    top_index = int(np.argmax(probabilities))

    label_map = {
        0: "WEAK",
        1: "MIXED",
        2: "CLEAN",
    }

    label = label_map[top_index]

    return {
        "label": label,
        "score": probability_to_score(probabilities),
        "strength": probability_to_strength(probabilities),
        "probabilities": {
            "WEAK": round(weak_prob, 4),
            "MIXED": round(mixed_prob, 4),
            "CLEAN": round(clean_prob, 4),
        },
    }


if __name__ == "__main__":
    print("calibrator.py loaded successfully.")