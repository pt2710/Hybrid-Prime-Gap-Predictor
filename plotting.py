"""
plotting.py

Module providing visualization utilities for prime-gap prediction experiments.
All functions save figures or tables to disk and close plots to free memory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_actual_vs_predicted(
    actual: np.ndarray,
    predicted: np.ndarray,
    output_path: str = "actual_vs_predicted.png"
) -> None:
    """
    Create and save a scatter plot comparing actual vs. predicted prime gaps.

    Parameters
    ----------
    actual : np.ndarray
        1D array of ground-truth prime gaps.
    predicted : np.ndarray
        1D array of model-predicted prime gaps.
    output_path : str, optional
        Filepath to save the generated PNG image (default: "actual_vs_predicted.png").

    Notes
    -----
    - Draws an identity line (y = x) to visualize perfect predictions.
    - Uses equal scaling on both axes to avoid distortion.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # Initialize figure with fixed aspect
    plt.figure(figsize=(8, 6))

    # Scatter plot of data points
    plt.scatter(actual, predicted, s=5, alpha=0.6, label="Data points")

    # Determine plot limits based on max values
    lim = max(np.nanmax(actual), np.nanmax(predicted))

    # Plot identity line for reference
    plt.plot([0, lim], [0, lim], linestyle='--', linewidth=1, color='gray', label="Ideal (y = x)")

    # Axis labels and title
    plt.xlabel("Actual Prime Gaps")
    plt.ylabel("Predicted Prime Gaps")
    plt.title("Actual vs. Predicted Prime Gaps")

    # Maintain square aspect ratio
    plt.gca().set_aspect('equal', 'box')

    # Display legend
    plt.legend()

    # Set axis limits to [0, lim]
    plt.xlim(0, lim)
    plt.ylim(0, lim)

    # Tidy layout and save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_gap_series(
    actual: np.ndarray,
    predicted: np.ndarray,
    output_path: str = "gap_time_series.png"
) -> None:
    """
    Save an overlaid time-series of actual vs. predicted prime gaps.

    Parameters
    ----------
    actual : np.ndarray
        Sequence of actual prime gaps indexed by prime order.
    predicted : np.ndarray
        Sequence of predicted gaps matching `actual` length.
    output_path : str, optional
        Filepath to save the time-series PNG (default: "gap_time_series.png").
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.figure(figsize=(10, 4))

    # X-axis is the index of each gap
    idx = np.arange(1, len(actual) + 1)

    # Plot actual and predicted series
    plt.plot(idx, actual, label="Actual Gaps", linewidth=1)
    plt.plot(idx, predicted, label="Predicted Gaps", linewidth=1, linestyle='--')

    # Labels and title
    plt.xlabel("Prime Index")
    plt.ylabel("Gap Size")
    plt.title("Prime Gap Time Series: Actual vs. Predicted")

    # Legend and layout
    plt.legend()
    plt.tight_layout()

    # Save and show figure (optional show)
    plt.savefig(output_path, dpi=300)
    # plt.show()  # uncomment if interactive display is desired
    plt.close()


def plot_primes_scatter(
    actual_primes: np.ndarray,
    predicted_primes: np.ndarray,
    output_path: str = "primes_scatter.png"
) -> None:
    """
    Create and save a scatter plot of actual vs. predicted prime values.

    Parameters
    ----------
    actual_primes : np.ndarray
        Array of true prime numbers.
    predicted_primes : np.ndarray
        Array of model-predicted prime values.
    output_path : str, optional
        Output PNG filepath (default: "primes_scatter.png").
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.figure(figsize=(8, 6))

    # Scatter actual vs. predicted primes
    plt.scatter(actual_primes, predicted_primes, s=5, alpha=0.6, label="Predictions")

    # Identity line for reference
    lim = max(np.nanmax(actual_primes), np.nanmax(predicted_primes))
    plt.plot([0, lim], [0, lim], linestyle='--', linewidth=1, color='gray', label="Ideal")

    # Formatting
    plt.xlabel("Actual Prime Value")
    plt.ylabel("Predicted Prime Value")
    plt.title("Actual vs. Predicted Prime Values")
    plt.gca().set_aspect('equal', 'box')
    plt.legend()
    plt.xlim(0, lim)
    plt.ylim(0, lim)
    plt.tight_layout()

    # Save and close
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_benchmark_table(
    benchmarks: dict,
    csv_path: str = "benchmark_times.csv"
) -> None:
    """
    Save benchmark timing results to a CSV file.

    Parameters
    ----------
    benchmarks : dict
        Mapping of method names to time per prime (in microseconds).
    csv_path : str, optional
        Path to output CSV file (default: "benchmark_times.csv").
    """
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    df = pd.DataFrame(
        list(benchmarks.items()),
        columns=["Method", "Time_per_prime_us"]
    )
    df.to_csv(csv_path, index=False)


def plot_benchmark(
    benchmarks: dict,
    output_path: str = "benchmark_bar_chart.png"
) -> None:
    """
    Create and save a bar chart of benchmark timings.

    Parameters
    ----------
    benchmarks : dict
        Map of method names to time per prime (μs).
    output_path : str, optional
        Filepath for the output PNG (default: "benchmark_bar_chart.png").
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    methods = list(benchmarks.keys())
    times = list(benchmarks.values())

    plt.figure(figsize=(6, 4))
    plt.bar(methods, times, edgecolor='black')
    plt.xticks(rotation=45, ha='right')

    plt.ylabel("Time per prime (μs)")
    plt.title("Prime-Finding Method Benchmark")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_spectral_field(
    x: np.ndarray,
    S_field: np.ndarray,
    piE: np.ndarray,
    output_path: str = "spectral_field.png"
) -> None:
    """
    Plot parity potential field S(x) and entropic wavefunction πE(x) in a two-panel figure.

    Parameters
    ----------
    x : np.ndarray
        Domain coordinate array.
    S_field : np.ndarray
        Parity potential values at each x.
    piE : np.ndarray
        Entropic wavefunction values corresponding to x.
    output_path : str, optional
        Destination PNG filepath (default: "spectral_field.png").
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Top panel: parity potential
    axs[0].plot(x, S_field, linewidth=1)
    axs[0].set_ylabel("Parity Potential S(x)")

    # Bottom panel: entropic wavefunction
    axs[1].plot(x, piE, linewidth=1)
    axs[1].set_xlabel("Domain Coordinate")
    axs[1].set_ylabel("Entropic Wavefunction πE(x)")

    fig.suptitle("Spectral Field and Entropic Wavefunction", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_prime_gap_comparison(
    prime_gaps: np.ndarray,
    h_best: np.ndarray,
    h_final: np.ndarray,
    predicted_gaps_test: np.ndarray,
    N: int,
    train_size: int,
    output_path: str = "prime_gap_comparison.png"
) -> None:
    """
    Plot comparison of actual prime gaps, baseline, tuned model, and test-set predictions.

    Parameters
    ----------
    prime_gaps : np.ndarray
        Full sequence of actual prime gaps up to N.
    h_best : np.ndarray
        Baseline model gaps for training portion.
    h_final : np.ndarray
        Domain-tuned model gaps for training portion.
    predicted_gaps_test : np.ndarray
        Model predictions for the test set.
    N : int
        Total number of primes considered.
    train_size : int
        Number of primes used for training.
    output_path : str, optional
        Output PNG filepath (default: "prime_gap_comparison.png").
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.figure(figsize=(15, 7))

    # X axes for train and test partitions
    x_train = np.arange(1, train_size + 1)
    x_test = np.arange(train_size + 1, N + 1)
    x_full = np.arange(1, N + 1)

    # Plot actual gaps for all primes
    plt.plot(x_full, prime_gaps, label="Actual prime gaps", linewidth=0.5)

    # Plot baseline and tuned on training segment
    plt.plot(x_train, h_best[:train_size], label="Baseline model", linewidth=0.5)
    plt.plot(x_train, h_final[:train_size], label="Domain-tuned model", linewidth=0.5, linestyle='--')

    # Plot test-set predictions
    plt.plot(x_test, predicted_gaps_test, label="Predicted gaps (test)", linewidth=0.5, linestyle=':')

    # Formatting
    plt.xlabel("Prime Index n")
    plt.ylabel("Gap Value")
    plt.title(f"Prime Gap Comparison up to N={N}, Train Size={train_size}")
    plt.legend()
    plt.tight_layout()

    # Save and close
    plt.savefig(output_path, dpi=300)
    plt.close()
