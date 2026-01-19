# Modified DNN solution to output top K candidate schedules matching PPO's TOPK count.
#
"""
Deep Neural Network approach for multi‑objective irrigation grouping optimization.

This patched version adds functionality to return the same number of top candidate
solutions as the PPO baseline.  After training the neural network on sampled
permutations, the script now optionally evaluates a set of high‑quality
permutations and reports the best `TOPK_RESULTS` solutions ranked by the composite
score (variance + margin penalty).  This allows direct comparison with the PPO
script, which outputs the top K solutions.

All original functionality is preserved; the new logic is contained in the
``main`` function and a modified return value from ``generate_training_data``.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional

try:
    # Import the evaluation environment from the existing project.  If this
    # fails because ``tree_evaluator.py`` is unavailable, please ensure that
    # module is installed on your PYTHONPATH.
    from ppo_env import IrrigationGroupingEnv
    from comparison_utils import (
        ComparisonConfig,
        build_environment as build_shared_env,
        composite_score,
        evaluate_order,
    )
except ImportError as e:
    raise ImportError(
        "Required modules not found. Make sure `ppo_env.py` and `comparison_utils.py` "
        "are accessible in the PYTHONPATH."
    ) from e


class SimpleNN:
    """A minimal two‑layer neural network implemented with NumPy.

    The network maps static feature vectors of shape (F,) to a scalar score.  A
    smaller score indicates that the lateral should be selected earlier.  The
    model consists of one hidden layer with a tanh activation followed by a
    linear output layer.  Mean squared error is used as the loss function.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int = 32, lr: float = 1e-3, seed: Optional[int] = None
    ) -> None:
        rng = np.random.default_rng(seed)
        # Xavier/Glorot initialisation for weights
        self.W1 = rng.normal(0, np.sqrt(2.0 / (input_dim + hidden_dim)), size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim, dtype=np.float64)
        self.W2 = rng.normal(0, np.sqrt(2.0 / (hidden_dim + 1)), size=(hidden_dim, 1))
        self.b2 = np.zeros(1, dtype=np.float64)
        self.lr = lr

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Forward pass through the network.

        Parameters
        ----------
        X : np.ndarray
            Input features of shape (N, F).

        Returns
        -------
        out : np.ndarray
            Output scores of shape (N,), squeezed from (N,1).
        cache : tuple
            Cached values for the backward pass.
        """
        z1 = X @ self.W1 + self.b1  # (N, hidden_dim)
        a1 = np.tanh(z1)  # (N, hidden_dim)
        z2 = a1 @ self.W2 + self.b2  # (N, 1)
        out = z2.squeeze()  # (N,)
        return out, (X, z1, a1)

    def backward(
        self, out: np.ndarray, y: np.ndarray, cache: Tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Backward pass computing gradients and updating the parameters.

        Parameters
        ----------
        out : np.ndarray
            Predicted outputs of shape (N,).
        y : np.ndarray
            Target outputs of shape (N,).
        cache : tuple
            Cached values from the forward pass.
        """
        X, z1, a1 = cache
        N = X.shape[0]
        # Compute gradient of mean squared error loss: dL/dout = 2 * (out - y) / N
        dloss = (out - y) * (2.0 / N)  # (N,)
        # Gradients for the output layer
        dW2 = a1.T @ dloss.reshape(-1, 1)  # (hidden_dim, 1)
        db2 = np.sum(dloss)  # scalar
        # Propagate to hidden layer
        da1 = dloss.reshape(-1, 1) @ self.W2.T  # (N, hidden_dim)
        dz1 = da1 * (1.0 - np.tanh(z1) ** 2)  # derivative of tanh
        # Gradients for the first layer
        dW1 = X.T @ dz1  # (input_dim, hidden_dim)
        db1 = np.sum(dz1, axis=0)  # (hidden_dim,)
        # Update parameters
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        verbose: bool = True,
    ) -> None:
        """Train the network using gradient descent.

        Parameters
        ----------
        X : np.ndarray
            Input features of shape (N, F).
        y : np.ndarray
            Target outputs of shape (N,).
        epochs : int
            Number of training epochs.
        verbose : bool
            Whether to print progress messages.
        """
        for epoch in range(epochs):
            out, cache = self.forward(X)
            loss = np.mean((out - y) ** 2)
            self.backward(out, y, cache)
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch + 1}/{epochs}, loss={loss:.6f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Compute the network output for input features.

        Parameters
        ----------
        X : np.ndarray
            Input features of shape (N, F).

        Returns
        -------
        np.ndarray
            Scores of shape (N,).
        """
        out, _ = self.forward(X)
        return out


def evaluate_permutation(env: IrrigationGroupingEnv, order: np.ndarray) -> Tuple[float, float]:
    """Evaluate a complete irrigation schedule using the environment.

    Parameters
    ----------
    env : IrrigationGroupingEnv
        Instance of the environment configured with the hydraulic evaluator and
        lateral mapping.
    order : np.ndarray
        Sequence of indices representing the order in which laterals are opened.

    Returns
    -------
    tuple(float, float)
        (final_variance, negative_min_margin).  ``final_variance`` is the
        variance of the minimum margins across groups at the end of the episode,
        and ``negative_min_margin`` is the negative of the smallest margin
        achieved during the episode (so that both objectives are minimised in
        optimisation).
    """
    metrics = evaluate_order(env, order.tolist())
    return metrics["final_var"], -metrics["min_margin"]


def generate_training_data(
    env: IrrigationGroupingEnv,
    num_samples: int = 600,
    top_k: int = 30,
    margin_weight: float = 1.0,
    seed: Optional[int] = None,
    return_top_permutations: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[List[np.ndarray]]]:
    """Generate training data by sampling random schedules.

    Randomly samples ``num_samples`` permutations of all laterals, evaluates each
    with the environment and selects the top ``top_k`` permutations based on a
    composite objective.  For each of the selected permutations, this function
    records the (lateral features, position) pairs which are used as training
    examples for the neural network.

    Parameters
    ----------
    env : IrrigationGroupingEnv
        Environment instance.
    num_samples : int
        Number of random permutations to generate.
    top_k : int
        Number of best permutations (lowest composite objective) to use for
        training.
    margin_weight : float
        Weight applied to the minimum-margin term in the composite score.
    seed : Optional[int]
        Random seed for reproducibility.
    return_top_permutations : bool
        Whether to return the list of top_k permutations in addition to training data.

    Returns
    -------
    X_train : np.ndarray
        Feature matrix of shape (top_k * N, F) where N is the number of
        laterals and F is the feature dimension.
    y_train : np.ndarray
        Target positions of shape (top_k * N,), normalised to [0,1].
    top_perms : Optional[List[np.ndarray]]
        If return_top_permutations=True, a list of the top_k permutation arrays.
    """
    rng = np.random.default_rng(seed)
    N = env.N
    # Use the static features from the environment as the input feature matrix
    X_static = env._feat_static.copy().astype(np.float64)
    samples: List[Tuple[np.ndarray, Tuple[float, float, float]]] = []
    for i in range(num_samples):
        perm = rng.permutation(N)
        var, neg_min = evaluate_permutation(env, perm)
        # Composite objective: prioritise low variance and high min margin
        obj = composite_score(var, -neg_min, margin_weight=margin_weight)
        samples.append((perm, (var, neg_min, obj)))
    # Sort by composite objective (lower is better)
    samples.sort(key=lambda x: x[1][2])
    # Select top_k
    top_samples = samples[:top_k]
    # Prepare training data
    X_list = []
    y_list = []
    for perm, (var, neg_min, obj) in top_samples:
        # For each position in the permutation, create a training example
        for pos, idx in enumerate(perm):
            X_list.append(X_static[idx])
            y_list.append(pos / float(N - 1))
    X_train = np.stack(X_list, axis=0)
    y_train = np.array(y_list, dtype=np.float64)
    if return_top_permutations:
        perms = [perm for perm, _ in top_samples]
        return X_train, y_train, perms
    return X_train, y_train, None


def build_environment(seed: int = 0, config: Optional[ComparisonConfig] = None) -> IrrigationGroupingEnv:
    """Instantiate the irrigation grouping environment with shared settings."""
    return build_shared_env(seed=seed, config=config)


def main() -> None:
    """Run the DNN training and evaluate the resulting schedule.

    This entry point has been extended to output the top ``TOPK_RESULTS`` candidate
    schedules (based on composite score) to match the number of solutions
    produced by the PPO evaluation script.
    """
    # Define how many solutions to output (e.g. match PPO's TOPK)
    TOPK_RESULTS = 30
    # Build environment
    config = ComparisonConfig()
    env = build_environment(seed=0, config=config)
    print(f"Loaded environment with {env.N} laterals and {env.F_static} static features.")
    # Generate training data and capture top permutations used for training
    X_train, y_train, top_perms = generate_training_data(
        env,
        num_samples=600,
        top_k=TOPK_RESULTS,
        margin_weight=1.0,
        seed=0,
        return_top_permutations=True,
    )
    print(f"Generated training dataset of {X_train.shape[0]} samples.")
    # Train neural network
    model = SimpleNN(input_dim=X_train.shape[1], hidden_dim=32, lr=1e-2, seed=0)
    model.fit(X_train, y_train, epochs=200, verbose=True)
    # Use model to predict ordering
    scores = model.predict(env._feat_static.astype(np.float64))
    # Lower scores should be selected earlier
    predicted_order = np.argsort(scores)
    # Evaluate predicted schedule
    final_var, neg_min_margin = evaluate_permutation(env, predicted_order)
    composite = composite_score(final_var, -neg_min_margin, margin_weight=1.0)
    print("Predicted schedule results:")
    print(f"  Final variance: {final_var:.6f}")
    print(f"  Minimum margin: {-neg_min_margin:.6f}")
    print(f"  Composite score: {composite:.6f}")
    # Optionally, print the predicted ordering and corresponding lateral IDs
    predicted_lids = [env.lateral_ids[i] for i in predicted_order]
    print("Predicted lateral ordering:")
    print(predicted_lids)
    # ----- New: Evaluate top candidate permutations and output top results -----
    if top_perms:
        print(f"\nTop {TOPK_RESULTS} candidate solutions by composite score:")
        candidate_records: List[Tuple[float, float, float, List[int]]] = []
        for perm in top_perms:
            var, neg_min = evaluate_permutation(env, perm)
            score = composite_score(var, -neg_min, margin_weight=1.0)
            candidate_records.append((score, var, -neg_min, perm.tolist()))
        candidate_records.sort(key=lambda x: x[0])
        for rank, (score, var, min_margin, perm) in enumerate(candidate_records, start=1):
            print(
                f"{rank:02d}: composite_score={score:.6f}, final_var={var:.6f}, "
                f"min_margin={min_margin:.6f}"
            )
            lids = [env.lateral_ids[i] for i in perm]
            print(f"    Ordering: {lids}")


if __name__ == "__main__":
    main()
