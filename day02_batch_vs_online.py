"""
day02_batch_vs_online.py
Goal: Simulate the structural difference between batch and online
learning loops using a trivial running-average "model" (no real ML
algorithm yet -- that starts Week 3). The point is the TRAINING LOOP
SHAPE, not the model itself.
"""

import numpy as np

# Simulate a stream of incoming data (e.g., daily average temperature
# readings), 30 "days" worth.
rng = np.random.default_rng(seed=42)
true_signal = 20 + 5 * np.sin(np.linspace(0, 3 * np.pi, 30))  # a slow trend
data_stream = true_signal + rng.normal(0, 1.5, size=30)        # + noise


def batch_train(data: np.ndarray) -> float:
    """
    Batch learning: must see ALL data before producing a model.
    Here, the 'model' is just the mean -- deliberately trivial, so we
    can focus purely on the *training loop shape*, not the algorithm.
    """
    return float(np.mean(data))


def online_train_step(current_estimate: float, new_point: float, learning_rate: float) -> float:
    """
    Online learning: update the model incrementally, one point at a
    time, using a learning rate -- structurally identical to the
    gradient descent update rule we'll derive properly in Week 3:
        estimate := estimate + learning_rate * (new_point - estimate)
    This IS a (very simplified) gradient descent step on a squared-error
    objective -- a preview, not a coincidence.
    """
    return current_estimate + learning_rate * (new_point - current_estimate)


if __name__ == "__main__":
    # --- Batch learning: retrain from scratch once all data is in ---
    batch_model = batch_train(data_stream)
    print(f"Batch model (trained once, on all 30 points): {batch_model:.2f}")

    # --- Online learning: two different learning rates, updated live ---
    for lr, label in [(0.5, "HIGH learning rate (fast, forgets easily)"),
                       (0.05, "LOW learning rate (slow, stable)")]:
        estimate = data_stream[0]  # initialize from the first point
        trajectory = [estimate]
        for point in data_stream[1:]:
            estimate = online_train_step(estimate, point, lr)
            trajectory.append(estimate)
        print(f"\nOnline model, {label}")
        print(f"  Final estimate after seeing all 30 points: {estimate:.2f}")
        print(f"  First 5 estimates: {[round(x, 2) for x in trajectory[:5]]}")