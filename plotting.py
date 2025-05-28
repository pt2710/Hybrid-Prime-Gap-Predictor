import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_actual_vs_predicted(actual: np.ndarray,
                             predicted: np.ndarray,
                             output_path: str = "actual_vs_predicted.png") -> None:
    """
    Scatter plot of actual vs. predicted prime gaps.
    Ensures identity line and equal axes for proper comparison.
    Saves a PNG to output_path.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(actual, predicted, s=5, alpha=0.6)
    # Identity line
    lim = max(np.nanmax(actual), np.nanmax(predicted))
    plt.plot([0, lim], [0, lim], linestyle='--', linewidth=1)
    # Labels and aspect
    plt.xlabel('Actual Prime Gaps')
    plt.ylabel('Predicted Prime Gaps')
    plt.title('Actual vs. Predicted Prime Gaps')
    plt.legend()
    plt.xlim(0, lim)
    plt.ylim(0, lim)
    plt.gca().set_aspect('equal', 'box')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_gap_series(actual: np.ndarray,
                    predicted: np.ndarray,
                    output_path: str = "gap_time_series.png") -> None:
    """
    Overlay time-series plot of actual vs. predicted prime gaps across prime index.
    Saves a PNG to output_path.
    """
    plt.figure(figsize=(10, 4))
    idx = np.arange(1, len(actual) + 1)
    plt.plot(idx, actual, label='Actual Gaps', linewidth=1)
    plt.plot(idx, predicted, label='Predicted Gaps', linewidth=1)
    plt.xlabel('Prime Index')
    plt.ylabel('Gap Size')
    plt.title('Prime Gaps: Actual vs. Predicted')
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()


def plot_primes_scatter(actual_primes: np.ndarray,
                         predicted_primes: np.ndarray,
                         output_path: str = "primes_scatter.png") -> None:
    """
    Scatter plot of actual vs. predicted prime values.
    Shows model's prime predictions against true primes.
    Ensures identity line and equal axes.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(actual_primes, predicted_primes, s=5, alpha=0.6)
    lim = max(np.nanmax(actual_primes), np.nanmax(predicted_primes))
    plt.plot([0, lim], [0, lim], linestyle='--', linewidth=1)
    plt.xlabel('Actual Prime Value')
    plt.ylabel('Predicted Prime Value')
    plt.title('Actual vs. Predicted Primes')
    plt.xlim(0, lim)
    plt.ylim(0, lim)
    plt.gca().set_aspect('equal', 'box')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_benchmark_table(benchmarks: dict,
                         csv_path: str = "benchmark_times.csv") -> None:
    """
    Save benchmark times per prime (in microseconds) to a CSV file.
    benchmarks: dict of {method_name: time_per_prime_us}
    """
    df = pd.DataFrame(list(benchmarks.items()), columns=["Method", "Time_per_prime_us"])
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    df.to_csv(csv_path, index=False)


def plot_benchmark(benchmarks: dict,
                   output_path: str = "benchmark_bar_chart.png") -> None:
    """
    Bar chart of benchmark times per prime.
    """
    methods = list(benchmarks.keys())
    times = list(benchmarks.values())
    plt.figure(figsize=(6, 4))
    plt.bar(methods, times)
    plt.ylabel('Time per prime (μs)')
    plt.title('Benchmark Comparison of Prime-Finding Methods')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_spectral_field(x: np.ndarray,
                        S_field: np.ndarray,
                        piE: np.ndarray,
                        output_path: str = "spectral_field.png") -> None:
    """
    Plot the spectral field S(x) and the entropic wavefunction piE(x).
    Produces a two-panel PNG saved to output_path.
    """
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axs[0].plot(x, S_field)
    axs[0].set_ylabel('Parity Potential S(x)')
    axs[0].set_title('Spectral Field: Parity Potential and Entropic Wavefunction')

    axs[1].plot(x, piE)
    axs[1].set_xlabel('Domain Coordinate')
    axs[1].set_ylabel('Entropic Wavefunction πE(x)')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_prime_gap_comparison(prime_gaps, h_best, h_final, predicted_gaps_test, N, train_size):
    plt.figure(figsize=(15, 7))
    x_train = np.arange(1, train_size + 1)
    x_test = np.arange(train_size + 1, N + 1)
    x_full = np.arange(1, N + 1)

    plt.plot(x_full, prime_gaps, label="Actual prime gaps", linewidth=0.5)
    plt.plot(x_train, h_best[:train_size], label="Baseline model", linewidth=0.5)  # Slice h_best
    plt.plot(x_train, h_final[:train_size], label="Domain-tuned model", linewidth=0.5, color='red')  # Slice h_final
    plt.plot(x_test, predicted_gaps_test, label="Predicted gaps (Test Set)", linewidth=0.5, color='green')
    plt.xlabel("Index n")
    plt.ylabel("Gap value")
    plt.title(f"Comparison of Actual and Predicted Prime Gaps for {N} Primes")
    plt.legend()
    plt.tight_layout()
    plt.close()