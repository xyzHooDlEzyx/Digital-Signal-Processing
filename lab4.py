import matplotlib.pyplot as plt
import numpy as np


# Variant 4 from the guide.
VARIANT = 4
FILTER_ORDER = 7
CUTOFF_FREQUENCY_HZ = 30.0
SAMPLING_FREQUENCY_HZ = 200.0
SAMPLING_INTERVAL_S = 1.0 / SAMPLING_FREQUENCY_HZ

# The final equation in the guide contains a_1 ... a_7 and no a_0 term.
# A leading zero therefore represents the absent x[n] term and keeps the
# highest delay equal to seven samples (a seventh-order FIR filter).
FOLLOW_GUIDE_FINAL_EQUATION = True


def lowpass_coefficient(k: int) -> float:
    """Return a_k from formula (7) of the guide."""
    if k == 0:
        return 2.0 * CUTOFF_FREQUENCY_HZ / SAMPLING_FREQUENCY_HZ

    normalized_argument = (
        2.0
        * np.pi
        * k
        * CUTOFF_FREQUENCY_HZ
        / SAMPLING_FREQUENCY_HZ
    )
    return (
        2.0
        * CUTOFF_FREQUENCY_HZ
        / SAMPLING_FREQUENCY_HZ
        * np.sin(normalized_argument)
        / normalized_argument
    )


coefficient_indices = np.arange(1, FILTER_ORDER + 1)
guide_coefficients = np.array(
    [lowpass_coefficient(int(k)) for k in coefficient_indices],
    dtype=float,
)
a0_limit_value = lowpass_coefficient(0)

if FOLLOW_GUIDE_FINAL_EQUATION:
    # y[n] = sum(a_k * x[n-k]), k = 1 ... 7.
    filter_coefficients = np.concatenate(([0.0], guide_coefficients))
else:
    # Mathematically complete k = 0 ... 7 version of formula (7).
    filter_coefficients = np.concatenate(([a0_limit_value], guide_coefficients))


def frequency_response(coefficients, point_count=4096):
    """Calculate H(f) directly, without scipy.signal.freqz."""
    frequencies_hz = np.linspace(
        0.0,
        SAMPLING_FREQUENCY_HZ / 2.0,
        point_count,
    )
    angular_frequency_rad_per_sample = (
        2.0 * np.pi * frequencies_hz / SAMPLING_FREQUENCY_HZ
    )
    delays = np.arange(len(coefficients))
    response = (
        np.exp(-1j * np.outer(angular_frequency_rad_per_sample, delays))
        @ coefficients
    )
    return frequencies_hz, response


def fir_filter_direct(signal, coefficients):
    """Evaluate the FIR difference equation sample by sample."""
    signal = np.asarray(signal, dtype=float)
    output = np.zeros_like(signal)

    for sample_index in range(len(signal)):
        for delay, coefficient in enumerate(coefficients):
            previous_index = sample_index - delay
            if previous_index >= 0:
                output[sample_index] += coefficient * signal[previous_index]

    return output


# Input x(t): the Fourier-series signal from laboratory work 2, variant 4.
LAB2_SAMPLES = np.array([0.2, 0.8, 0.6, 0.4, 0.2, 0.8], dtype=float)
LAB2_OMEGA_0_RAD_S = 1.0
LAB2_PERIOD_S = 2.0 * np.pi / LAB2_OMEGA_0_RAD_S
lab2_sample_numbers = np.arange(1, len(LAB2_SAMPLES) + 1, dtype=float)
lab2_sample_times = (
    lab2_sample_numbers * LAB2_PERIOD_S / len(LAB2_SAMPLES)
)

lab2_a0 = 2.0 * np.mean(LAB2_SAMPLES)
lab2_harmonics = np.arange(1, len(LAB2_SAMPLES) // 2, dtype=int)
lab2_cosine_coefficients = np.array(
    [
        2.0
        / len(LAB2_SAMPLES)
        * np.sum(
            LAB2_SAMPLES
            * np.cos(k * LAB2_OMEGA_0_RAD_S * lab2_sample_times)
        )
        for k in lab2_harmonics
    ]
)
lab2_sine_coefficients = np.array(
    [
        2.0
        / len(LAB2_SAMPLES)
        * np.sum(
            LAB2_SAMPLES
            * np.sin(k * LAB2_OMEGA_0_RAD_S * lab2_sample_times)
        )
        for k in lab2_harmonics
    ]
)
lab2_nyquist_harmonic = len(LAB2_SAMPLES) // 2
lab2_nyquist_coefficient = (
    1.0
    / len(LAB2_SAMPLES)
    * np.sum(
        LAB2_SAMPLES
        * np.cos(
            lab2_nyquist_harmonic
            * LAB2_OMEGA_0_RAD_S
            * lab2_sample_times
        )
    )
)


def lab2_fourier_signal(time_values):
    """Evaluate the periodic Fourier interpolation from laboratory work 2."""
    time_values = np.asarray(time_values, dtype=float)
    result = np.full_like(time_values, lab2_a0 / 2.0)

    for index, harmonic in enumerate(lab2_harmonics):
        result += lab2_cosine_coefficients[index] * np.cos(
            harmonic * LAB2_OMEGA_0_RAD_S * time_values
        )
        result += lab2_sine_coefficients[index] * np.sin(
            harmonic * LAB2_OMEGA_0_RAD_S * time_values
        )

    result += lab2_nyquist_coefficient * np.cos(
        lab2_nyquist_harmonic * LAB2_OMEGA_0_RAD_S * time_values
    )
    return result


def format_difference_equation(coefficients):
    """Create a readable equation from the nonzero FIR coefficients."""
    terms = []
    for delay, coefficient in enumerate(coefficients):
        if np.isclose(coefficient, 0.0):
            continue
        variable = "x[n]" if delay == 0 else f"x[n-{delay}]"
        magnitude = abs(coefficient)
        if not terms:
            terms.append(f"{coefficient:.6f} {variable}")
        else:
            sign = "+" if coefficient >= 0 else "-"
            terms.append(f" {sign} {magnitude:.6f} {variable}")
    return "y[n] = " + "".join(terms)


def main():
    frequencies_hz, response = frequency_response(filter_coefficients)
    amplitude_response = np.abs(response)
    phase_response_degrees = np.degrees(np.unwrap(np.angle(response)))

    # Two periods of the signal from laboratory work 2, sampled at 200 Hz.
    time_s = np.arange(
        0.0,
        2.0 * LAB2_PERIOD_S,
        SAMPLING_INTERVAL_S,
    )
    input_signal = lab2_fourier_signal(time_s)
    output_signal = fir_filter_direct(input_signal, filter_coefficients)

    # Independent convolution check for the implemented difference equation.
    convolution_output = np.convolve(
        input_signal,
        filter_coefficients,
        mode="full",
    )[: len(input_signal)]
    validation_error = np.max(np.abs(output_signal - convolution_output))

    print("LABORATORY WORK 4. VARIANT 4")
    print("Non-recursive asymmetric low-pass FIR filter")
    print("=" * 72)
    print(f"Filter order N = {FILTER_ORDER}")
    print(f"Cutoff frequency f_g = {CUTOFF_FREQUENCY_HZ:.1f} Hz")
    print(f"Sampling frequency f_a = {SAMPLING_FREQUENCY_HZ:.1f} Hz")
    print(f"Sampling interval T_a = {SAMPLING_INTERVAL_S:.6f} s")
    print(
        "Normalized cutoff f_g/(f_a/2) = "
        f"{CUTOFF_FREQUENCY_HZ / (SAMPLING_FREQUENCY_HZ / 2):.3f}"
    )

    print("\nCoefficients from formula (7):")
    print(f"a_0 (limit at k=0) = {a0_limit_value:.9f}")
    for k, coefficient in zip(coefficient_indices, guide_coefficients):
        print(f"a_{k} = {coefficient:+.9f}")

    print("\nEquation used (the final equation printed in the guide):")
    print(format_difference_equation(filter_coefficients))
    print(
        "\nMethodical note: formula (7) gives a_0 = 0.3, but the guide's "
        "final equation omits x[n]. The program follows that printed equation."
    )
    print(f"Direct-equation/convolution validation error = {validation_error:.3e}")

    figure, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(frequencies_hz, amplitude_response, color="tab:blue")
    axes[0, 0].axvline(
        CUTOFF_FREQUENCY_HZ,
        color="tab:red",
        linestyle="--",
        label=r"$f_g=30$ Hz",
    )
    axes[0, 0].set_title("Amplitude-frequency response")
    axes[0, 0].set_xlabel("Frequency, Hz")
    axes[0, 0].set_ylabel(r"$|H(f)|$")
    axes[0, 0].legend()

    axes[0, 1].plot(
        frequencies_hz,
        phase_response_degrees,
        color="tab:orange",
    )
    axes[0, 1].axvline(
        CUTOFF_FREQUENCY_HZ,
        color="tab:red",
        linestyle="--",
    )
    axes[0, 1].set_title("Phase-frequency response")
    axes[0, 1].set_xlabel("Frequency, Hz")
    axes[0, 1].set_ylabel("Phase, degrees")

    markerline, stemlines, baseline = axes[1, 0].stem(
        np.arange(len(filter_coefficients)),
        filter_coefficients,
    )
    plt.setp(markerline, color="tab:red", markersize=7)
    plt.setp(stemlines, color="tab:blue", linewidth=1.5)
    plt.setp(baseline, color="black", linewidth=0.8)
    axes[1, 0].set_title("Перехідна імпульсна характеристика")
    axes[1, 0].set_xlabel("Номер відліку k")
    axes[1, 0].set_ylabel(r"$h[k]$")

    axes[1, 1].plot(
        time_s,
        input_signal,
        label="input signal x(t)",
        linewidth=1.6,
    )
    axes[1, 1].plot(
        time_s,
        output_signal,
        label="output signal y(t)",
        linewidth=1.6,
    )
    axes[1, 1].set_title("Output response of the filter to the signal from lab #2")
    axes[1, 1].set_xlabel("time, s")
    axes[1, 1].set_ylabel("Amplitude")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.grid(True, linestyle=":", alpha=0.65)

    figure.suptitle(
        "Laboratory work #4 — variant 4",
        fontsize=15,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    plt.show()


if __name__ == "__main__":
    main()
