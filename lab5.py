from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

try:
    import pywt
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "PyWavelets is not installed. Run: python3 -m pip install PyWavelets"
    ) from error


VARIANT = 4
WAVELET = "db4"
LEVEL = 5
THRESHOLD_RULE = "minimaxi"
THRESHOLD_MODE = "hard"
RESCALING = "type 1"


def load_signal(path: Path) -> tuple[np.ndarray, float]:
    """Load the signal and sampling frequency from the MATLAB file."""
    data = loadmat(path)
    if "signal" not in data:
        available = [key for key in data if not key.startswith("__")]
        raise KeyError(f"Variable 'signal' was not found. Available variables: {available}")

    signal = np.asarray(data["signal"], dtype=float).squeeze()
    if signal.ndim != 1 or signal.size == 0:
        raise ValueError("Variable 'signal' must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(signal)):
        raise ValueError("The signal contains NaN or infinite values.")

    sampling_frequency = 1.0
    if "Fd" in data:
        sampling_frequency = float(np.asarray(data["Fd"]).squeeze())
    elif "Fs" in data:
        sampling_frequency = float(np.asarray(data["Fs"]).squeeze())
    return signal, sampling_frequency


def estimate_noise_sigma(c_d1: np.ndarray) -> float:
    """Type-1 noise estimate from first-level detail coefficients."""
    return float(np.median(np.abs(c_d1)) / 0.6745)


def minimaxi_threshold(vector_length: int, sigma: float) -> float:
    """MATLAB minimaxi threshold from the formula in the lab guide."""
    if vector_length <= 32:
        return 0.0
    normalized_threshold = 0.3936 + 0.1829 * np.log2(vector_length)
    return float(sigma * normalized_threshold)


    # wavedec returns [cA5, cD5, cD4, cD3, cD2, cD1].
    coefficients = pywt.wavedec(signal, WAVELET, level=LEVEL, mode="symmetric")
    approximation = coefficients[0]
    details = coefficients[1:]

    sigma = estimate_noise_sigma(details[-1])
    thresholds = [minimaxi_threshold(detail.size, sigma) for detail in details]
    thresholded_details = [
        pywt.threshold(detail, threshold, mode=THRESHOLD_MODE)
        for detail, threshold in zip(details, thresholds)
    ]

    reconstructed = pywt.waverec(
        [approximation, *thresholded_details], WAVELET, mode="symmetric"
    )[: signal.size]

    return reconstructed, coefficients, thresholded_details, sigma, thresholds


def calculate_metrics(signal, denoised, details, thresholded_details):
    residual = signal - denoised
    residual_rms = float(np.sqrt(np.mean(residual**2)))
    denoised_rms = float(np.sqrt(np.mean(denoised**2)))
    estimated_snr = (
        float(20 * np.log10(denoised_rms / residual_rms))
        if residual_rms > 0
        else float("inf")
    )
    zeroed = [
        int(np.count_nonzero(before) - np.count_nonzero(after))
        for before, after in zip(details, thresholded_details)
    ]
    return {
        "samples": int(signal.size),
        "original_mean": float(np.mean(signal)),
        "original_std": float(np.std(signal)),
        "denoised_mean": float(np.mean(denoised)),
        "denoised_std": float(np.std(denoised)),
        "residual_rms": residual_rms,
        "estimated_snr_db": estimated_snr,
        "zeroed_coefficients": zeroed,
    }


def plot_signal_results(signal, denoised, sampling_frequency):
    time = np.arange(signal.size) / sampling_frequency
    residual = signal - denoised

    figure, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(time, signal, color="tab:blue", linewidth=0.9)
    axes[0].set_title("Noisy signal")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(time, denoised, color="tab:green", linewidth=1.15)
    axes[1].set_title("Signal after wavelet denoising")
    axes[1].set_ylabel("Amplitude")

    axes[2].plot(time, residual, color="tab:red", linewidth=0.8)
    axes[2].set_title("Removed noise component")
    axes[2].set_xlabel("Time, s")
    axes[2].set_ylabel("Amplitude")

    for axis in axes:
        axis.grid(True, linestyle=":", alpha=0.65)
    figure.suptitle("Laboratory work 5, variant 4: db4, level 5", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.96))


def plot_coefficients(coefficients, thresholded_details, thresholds):
    details = coefficients[1:]
    figure, axes = plt.subplots(LEVEL, 2, figsize=(14, 12))
    levels = list(range(LEVEL, 0, -1))

    for row, (level, before, after, threshold) in enumerate(
        zip(levels, details, thresholded_details, thresholds)
    ):
        axes[row, 0].stem(before, linefmt="tab:blue", markerfmt=" ", basefmt=" ")
        axes[row, 0].axhline(threshold, color="tab:red", linestyle="--", linewidth=0.8)
        axes[row, 0].axhline(-threshold, color="tab:red", linestyle="--", linewidth=0.8)
        axes[row, 0].set_title(f"cD{level} before thresholding; τ = {threshold:.6f}")

        axes[row, 1].stem(after, linefmt="tab:green", markerfmt=" ", basefmt=" ")
        axes[row, 1].set_title(f"cD{level} after hard thresholding")

        for axis in axes[row]:
            axis.grid(True, linestyle=":", alpha=0.55)
            axis.set_ylabel("Coefficient")

    axes[-1, 0].set_xlabel("Coefficient index")
    axes[-1, 1].set_xlabel("Coefficient index")
    figure.tight_layout()


def print_metrics(source, sampling_frequency, sigma, thresholds, metrics):
    levels = list(range(LEVEL, 0, -1))
    lines = [
        "LABORATORY WORK 5. VARIANT 4",
        f"Input file: {source}",
        f"Number of samples: {metrics['samples']}",
        f"Sampling frequency: {sampling_frequency:.6f} Hz",
        f"Wavelet: {WAVELET}",
        f"Decomposition level: {LEVEL}",
        f"Threshold selection rule: {THRESHOLD_RULE}",
        f"Thresholding mode: {THRESHOLD_MODE}",
        f"Threshold rescaling: {RESCALING}",
        f"Estimated noise standard deviation sigma: {sigma:.9f}",
        "",
        "THRESHOLDS AND ZEROED COEFFICIENTS",
    ]
    for level, threshold, zeroed in zip(
        levels, thresholds, metrics["zeroed_coefficients"]
    ):
        lines.append(
            f"cD{level}: threshold = {threshold:.9f}; "
            f"zeroed coefficients = {zeroed}"
        )
    lines.extend(
        [
            "",
            "SIGNAL STATISTICS",
            f"Original signal mean: {metrics['original_mean']:.9f}",
            f"Original signal standard deviation: {metrics['original_std']:.9f}",
            f"Denoised signal mean: {metrics['denoised_mean']:.9f}",
            f"Denoised signal standard deviation: {metrics['denoised_std']:.9f}",
            f"Removed component RMS: {metrics['residual_rms']:.9f}",
            f"Estimated signal-to-removed-component ratio: "
            f"{metrics['estimated_snr_db']:.6f} dB",
        ]
    )
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Path to the .mat file (default: Lab_3_4.mat next to lab5.py)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    source = Path(args.input) if args.input else script_dir / "Lab_3_4.mat"
    if not source.exists():
        raise FileNotFoundError(
            f"Input file was not found: {source}\n"
            "Place Lab_3_4.mat next to lab5.py or pass its path as an argument."
        )
    signal, sampling_frequency = load_signal(source)
    denoised, coefficients, thresholded_details, sigma, thresholds = denoise_signal(signal)
    metrics = calculate_metrics(signal, denoised, coefficients[1:], thresholded_details)

    print_metrics(source.name, sampling_frequency, sigma, thresholds, metrics)
    plot_signal_results(signal, denoised, sampling_frequency)
    plot_coefficients(coefficients, thresholded_details, thresholds)
    plt.show()


if __name__ == "__main__":
    main()
