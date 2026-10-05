import numpy as np
import matplotlib.pyplot as plt


# Laboratory work 3: Discrete Fourier Transform
# Variant 4
VARIANT = 4
T_A = 0.1  # Sampling interval, seconds
F_A = 1 / T_A  # Sampling frequency, Hz

n = np.arange(10)
t = n * T_A
signal = np.array(
    [0.0, -3.0, -7.0, -2.0, 0.3, 2.0, 0.2, -0.8, -0.2, 0.0],
    dtype=float,
)
N = len(signal)
T_OBSERVATION = N * T_A
DELTA_F = 1 / T_OBSERVATION


def direct_dft(values):
    """Calculate the forward DFT directly from its definition."""
    values = np.asarray(values, dtype=complex)
    sample_count = len(values)
    spectrum = np.zeros(sample_count, dtype=complex)

    for k in range(sample_count):
        for sample_index in range(sample_count):
            angle = -2j * np.pi * k * sample_index / sample_count
            spectrum[k] += values[sample_index] * np.exp(angle)

    return spectrum


def inverse_dft(spectrum):
    """Calculate the inverse DFT directly from its definition."""
    spectrum = np.asarray(spectrum, dtype=complex)
    sample_count = len(spectrum)
    values = np.zeros(sample_count, dtype=complex)

    for sample_index in range(sample_count):
        for k in range(sample_count):
            angle = 2j * np.pi * k * sample_index / sample_count
            values[sample_index] += spectrum[k] * np.exp(angle)

    return values / sample_count


# Manual DFT and NumPy FFT. The guide uses F(jw_k) = T_a * DFT[k]
# when approximating the continuous Fourier transform.
dft_manual = direct_dft(signal)
dft_numpy = np.fft.fft(signal)
continuous_spectrum = T_A * dft_manual

frequencies = np.fft.fftfreq(N, d=T_A)
angular_frequencies = 2 * np.pi * frequencies

amplitude_dft = np.abs(dft_manual)
amplitude_continuous = np.abs(continuous_spectrum)
phase_degrees = np.degrees(np.angle(dft_manual))

# Put zero frequency in the center, as required by the assignment.
frequencies_centered = np.fft.fftshift(frequencies)
angular_frequencies_centered = 2 * np.pi * frequencies_centered
dft_centered = np.fft.fftshift(dft_manual)
continuous_centered = T_A * dft_centered
amplitude_centered = np.abs(continuous_centered)
phase_centered = np.degrees(np.angle(dft_centered))

# Numerical validation.
fft_difference = np.max(np.abs(dft_manual - dft_numpy))
reconstructed = inverse_dft(dft_manual)
reconstruction_error = np.max(np.abs(reconstructed.real - signal))
imaginary_reconstruction_error = np.max(np.abs(reconstructed.imag))

energy_time = np.sum(np.abs(signal) ** 2)
energy_frequency = np.sum(np.abs(dft_manual) ** 2) / N
parseval_error = abs(energy_time - energy_frequency)


def print_unshifted_spectrum():
    print("\nUNSHIFTED SPECTRUM")
    print(
        " k |  f_k, Hz | omega_k, rad/s |       Re{X[k]} |       Im{X[k]} "
        "|    |X[k]| | T_a|X[k]| | phase, deg"
    )
    print("-" * 112)

    for k in range(N):
        print(
            f"{k:2d} | {frequencies[k]:8.3f} | {angular_frequencies[k]:14.6f} "
            f"| {dft_manual[k].real:14.6f} | {dft_manual[k].imag:14.6f} "
            f"| {amplitude_dft[k]:9.6f} | {amplitude_continuous[k]:10.6f} "
            f"| {phase_degrees[k]:10.3f}"
        )


def print_centered_spectrum():
    print("\nCENTERED AMPLITUDE SPECTRUM")
    print(" f, Hz | omega, rad/s | T_a|X(f)|")
    print("-" * 42)
    for frequency, omega, amplitude in zip(
        frequencies_centered,
        angular_frequencies_centered,
        amplitude_centered,
    ):
        print(f"{frequency:6.1f} | {omega:12.6f} | {amplitude:10.6f}")


if __name__ == "__main__":
    print("LABORATORY WORK 3. DISCRETE FOURIER TRANSFORM")
    print(f"Variant: {VARIANT}")
    print("=" * 70)
    print(f"Sampling interval T_a = {T_A:.3f} s")
    print(f"Sampling frequency f_a = {F_A:.3f} Hz")
    print(f"Number of samples N = {N}")
    print(f"Observation time N*T_a = {T_OBSERVATION:.3f} s")
    print(f"Frequency resolution Delta_f = {DELTA_F:.3f} Hz")
    print(f"Nyquist frequency = {F_A / 2:.3f} Hz")
    print(f"Samples: {signal}")

    print_unshifted_spectrum()
    print_centered_spectrum()

    print("\nVALIDATION")
    print(f"Maximum |manual DFT - numpy FFT| = {fft_difference:.3e}")
    print(f"Maximum inverse-DFT reconstruction error = {reconstruction_error:.3e}")
    print(
        "Maximum imaginary part after inverse DFT = "
        f"{imaginary_reconstruction_error:.3e}"
    )
    print(f"Signal energy in time domain = {energy_time:.9f}")
    print(f"Signal energy from spectrum = {energy_frequency:.9f}")
    print(f"Parseval identity error = {parseval_error:.3e}")

    # Graphs corresponding to the guide: signal, real part, imaginary part,
    # centered amplitude spectrum, and centered phase spectrum.
    figure, axes = plt.subplots(3, 2, figsize=(13, 12))

    markerline, stemlines, baseline = axes[0, 0].stem(t, signal)
    plt.setp(markerline, color="red", markersize=6)
    plt.setp(stemlines, color="tab:blue", linewidth=1.5)
    axes[0, 0].set_title("Sampled signal")
    axes[0, 0].set_xlabel("t, s")
    axes[0, 0].set_ylabel("f(nT_a)")

    axes[0, 1].stem(frequencies_centered, dft_centered.real)
    axes[0, 1].set_title("Real part of centered DFT")
    axes[0, 1].set_xlabel("f, Hz")
    axes[0, 1].set_ylabel("Re{X(f)}")

    axes[1, 0].stem(frequencies_centered, dft_centered.imag)
    axes[1, 0].set_title("Imaginary part of centered DFT")
    axes[1, 0].set_xlabel("f, Hz")
    axes[1, 0].set_ylabel("Im{X(f)}")

    axes[1, 1].stem(frequencies_centered, amplitude_centered)
    axes[1, 1].set_title("Centered amplitude spectrum")
    axes[1, 1].set_xlabel("f, Hz")
    axes[1, 1].set_ylabel("T_a |X(f)|")

    significant = amplitude_centered > 1e-12
    axes[2, 0].stem(
        frequencies_centered[significant],
        phase_centered[significant],
    )
    axes[2, 0].set_title("Centered phase spectrum")
    axes[2, 0].set_xlabel("f, Hz")
    axes[2, 0].set_ylabel("Phase, degrees")

    axes[2, 1].axis("off")
    axes[2, 1].text(
        0.03,
        0.95,
        "Variant 4 summary\n\n"
        f"T_a = {T_A:.1f} s\n"
        f"f_a = {F_A:.1f} Hz\n"
        f"N T_a = {T_OBSERVATION:.1f} s\n"
        f"Delta f = {DELTA_F:.1f} Hz\n"
        f"f_Nyquist = {F_A / 2:.1f} Hz\n\n"
        f"Manual/FFT error: {fft_difference:.2e}\n"
        f"Reconstruction error: {reconstruction_error:.2e}\n"
        f"Parseval error: {parseval_error:.2e}",
        va="top",
        fontsize=12,
    )

    for axis in axes.flat:
        if axis.axison:
            axis.grid(True, linestyle=":", alpha=0.65)

    figure.suptitle(
        "Laboratory work 3: DFT spectrum, variant 4",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()
