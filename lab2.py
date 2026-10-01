import numpy as np
import matplotlib.pyplot as plt


VARIANT = 4

n = np.arange(1, 7, dtype=float)
y = np.array([0.2, 0.8, 0.6, 0.4, 0.2, 0.8], dtype=float)
N = len(y)


omega_0 = 1.0
T_0 = 2 * np.pi / omega_0
T_a = T_0 / N
t = n * T_a

a0 = (2 / N) * np.sum(y)

harmonics = np.arange(1, N // 2, dtype=int)
a = np.array(
    [(2 / N) * np.sum(y * np.cos(k * omega_0 * t)) for k in harmonics]
)
b = np.array(
    [(2 / N) * np.sum(y * np.sin(k * omega_0 * t)) for k in harmonics]
)

nyquist_harmonic = N // 2
c_nyquist = (1 / N) * np.sum(
    y * np.cos(nyquist_harmonic * omega_0 * t)
)


def fourier_approximation(t_values):
    """Evaluate the real Fourier interpolation of the six samples."""
    t_values = np.asarray(t_values, dtype=float)
    result = np.full_like(t_values, a0 / 2, dtype=float)

    for index, k in enumerate(harmonics):
        result += a[index] * np.cos(k * omega_0 * t_values)
        result += b[index] * np.sin(k * omega_0 * t_values)

    result += c_nyquist * np.cos(
        nyquist_harmonic * omega_0 * t_values
    )
    return result


x = 0.1 * VARIANT * n
log_y = np.log(y)

sum_x = np.sum(x)
sum_x2 = np.sum(x**2)
sum_log_y = np.sum(log_y)
sum_x_log_y = np.sum(x * log_y)

B = (N * sum_x_log_y - sum_x * sum_log_y) / (N * sum_x2 - sum_x**2)
A = (sum_log_y - B * sum_x) / N


def exponential_approximation(x_values):
    """Evaluate the exponential least-squares approximation."""
    return np.exp(A + B * np.asarray(x_values, dtype=float))



B_numpy, A_numpy = np.polyfit(x, log_y, 1)


if __name__ == "__main__":
    print("LABORATORY WORK 2. VARIANT 4")
    print("_" * 70)

    print("\nTASK 1: FOURIER-SERIES APPROXIMATION")
    print(f"Number of samples N = {N}")
    print(f"Fundamental frequency omega_0 = {omega_0:.6f} rad/s")
    print(f"Signal period T_0 = {T_0:.6f}")
    print(f"Sampling interval T_a = {T_a:.6f}")
    print(f"a_0 = {a0:.6f}")

    for index, k in enumerate(harmonics):
        print(f"a_{k} = {a[index]: .6f}; b_{k} = {b[index]: .6f}")

    print(
        f"c_{nyquist_harmonic} = {c_nyquist: .6f} "
        "(Nyquist cosine coefficient)"
    )

    fourier_at_nodes = fourier_approximation(t)
    fourier_error = np.max(np.abs(fourier_at_nodes - y))
    print("\nValues reconstructed at the sample points:")
    for sample_number, time_value, measured, reconstructed in zip(
        n.astype(int), t, y, fourier_at_nodes
    ):
        print(
            f"n={sample_number}: t={time_value:.6f}, "
            f"y={measured:.6f}, f_R(t)={reconstructed:.6f}"
        )
    print(f"Maximum reconstruction error = {fourier_error:.3e}")

    print("\nTASK 2: EXPONENTIAL APPROXIMATION")
    print(f"x = {x}")
    print(f"ln(y) = {log_y}")
    print(f"Manual coefficients: A = {A:.9f}, B = {B:.9f}")
    print(
        f"NumPy coefficients: A = {A_numpy:.9f}, "
        f"B = {B_numpy:.9f}"
    )
    print(f"|A - A_numpy| = {abs(A - A_numpy):.3e}")
    print(f"|B - B_numpy| = {abs(B - B_numpy):.3e}")
    print(f"f(x) = exp({A:.6f} {B:+.6f}x)")
    print(f"f(x) = {np.exp(A):.6f} * exp({B:.6f}x)")

    exponential_at_nodes = exponential_approximation(x)
    exponential_rmse = np.sqrt(np.mean((y - exponential_at_nodes) ** 2))
    print(f"RMSE in the original y scale = {exponential_rmse:.6f}")


    t_plot = np.linspace(0, 2 * T_0, 1000)
    y_fourier_plot = fourier_approximation(t_plot)


    t_samples_two_periods = np.concatenate((t, t + T_0))
    y_samples_two_periods = np.tile(y, 2)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    axes[0].plot(
        t_plot,
        y_fourier_plot,
        color="blue",
        linewidth=2,
        label="Fourier approximation",
    )
    axes[0].scatter(
        t_samples_two_periods,
        y_samples_two_periods,
        color="red",
        s=55,
        zorder=5,
        label="Input samples",
    )
    axes[0].axvline(T_0, color="gray", linestyle="--", alpha=0.7)
    axes[0].set_title("Periodic signal approximation")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("f_R(t)")
    axes[0].grid(linestyle=":", alpha=0.7)
    axes[0].legend()

    # Plot 2: exponential approximation.
    x_plot = np.linspace(x[0], x[-1], 500)
    y_exponential_plot = exponential_approximation(x_plot)

    axes[1].scatter(
        x,
        y,
        color="red",
        s=55,
        zorder=5,
        label="Input samples",
    )
    axes[1].plot(
        x_plot,
        y_exponential_plot,
        color="green",
        linewidth=2,
        label=f"f(x) = exp({A:.3f} {B:+.3f}x)",
    )
    axes[1].set_title("Exponential approximation")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("f(x)")
    axes[1].grid(linestyle=":", alpha=0.7)
    axes[1].legend()

    fig.suptitle("Laboratory work 2, variant 4", fontsize=14)
    fig.tight_layout()
    plt.show()
