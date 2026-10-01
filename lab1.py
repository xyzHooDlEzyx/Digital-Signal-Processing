import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

x = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
y = np.array([1, 2, -4, -6, 8, 10, 12, 10], dtype=float)

N = len(x)

assert len(x) == len(y) == 8
assert np.all(np.diff(x) > 0)


sum_x = np.sum(x)
sum_y = np.sum(y)
sum_x2 = np.sum(x ** 2)
sum_x3 = np.sum(x ** 3)
sum_x4 = np.sum(x ** 4)
sum_xy = np.sum(x * y)
sum_x2y = np.sum((x ** 2) * y)

matrix = np.array([[N, sum_x, sum_x2],
                   [sum_x, sum_x2, sum_x3],
                   [sum_x2, sum_x3, sum_x4]])

rhs = np.array([sum_y, sum_xy, sum_x2y])

delta = np.linalg.det(matrix)

matrix_a = matrix.copy()
matrix_a[:, 0] = rhs
delta_a = np.linalg.det(matrix_a)

matrix_b = matrix.copy()
matrix_b[:, 1] = rhs
delta_b = np.linalg.det(matrix_b)

matrix_c = matrix.copy()
matrix_c[:, 2] = rhs
delta_c = np.linalg.det(matrix_c)

a = delta_a / delta
b = delta_b / delta
c = delta_c / delta

p_x = f"{a:.4f} {b:+.4f}x {c:+.4f}x^2"

polyfit_coeffs = np.polyfit(x, y, 2)
c_np, b_np, a_np = polyfit_coeffs



h = np.diff(x)
inner_nodes = N - 2

spline_matrix = np.zeros((inner_nodes, inner_nodes))
spline_rhs = np.zeros(inner_nodes)

for row in range(inner_nodes):
    node = row + 1
    h_left = h[node - 1]
    h_right = h[node]

    if row > 0:
        spline_matrix[row, row - 1] = h_left

    spline_matrix[row, row] = 2 * (h_left + h_right)

    if row < inner_nodes - 1:
        spline_matrix[row, row + 1] = h_right

    spline_rhs[row] = 6 * (
        (y[node + 1] - y[node]) / h_right
        - (y[node] - y[node - 1]) / h_left
    )


second_derivatives = np.zeros(N)
second_derivatives[1:-1] = np.linalg.solve(spline_matrix, spline_rhs)


spline_a = y[:-1].copy()
spline_b = (
    np.diff(y) / h
    - h * (2 * second_derivatives[:-1] + second_derivatives[1:]) / 6
)
spline_c = second_derivatives[:-1] / 2
spline_d = np.diff(second_derivatives) / (6 * h)


def evaluate_manual_spline(x_values):
    """Evaluate the manually calculated piecewise natural cubic spline."""
    x_values = np.asarray(x_values, dtype=float)
    interval = np.searchsorted(x, x_values, side="right") - 1
    interval = np.clip(interval, 0, N - 2)
    dx = x_values - x[interval]

    return (
        spline_a[interval]
        + spline_b[interval] * dx
        + spline_c[interval] * dx**2
        + spline_d[interval] * dx**3
    )



scipy_spline = CubicSpline(x, y, bc_type="natural")

if __name__ == "__main__":
    print("TASK 1: QUADRATIC APPROXIMATION")
    print(f"Polynomial coefficients: a = {a}, b = {b}, c = {c}")
    print(f"Polynomial from numpy: a = {a_np}, b = {b_np}, c = {c_np}")
    print(f"Polynomial equation: y = {p_x}")

    print("\nTASK 2: NATURAL CUBIC SPLINE")
    print("Spline system matrix:")
    print(spline_matrix)
    print("Spline right-hand side:")
    print(spline_rhs)
    print("Second derivatives at the nodes:")
    print(second_derivatives)

    print("\nPiecewise spline coefficients:")
    for i in range(N - 1):
        print(
            f"[{x[i]:g}, {x[i + 1]:g}]: "
            f"S_{i + 1}(x) = {spline_a[i]:.6f} "
            f"{spline_b[i]:+.6f}(x-{x[i]:g}) "
            f"{spline_c[i]:+.6f}(x-{x[i]:g})^2 "
            f"{spline_d[i]:+.6f}(x-{x[i]:g})^3"
        )

    x_fit = np.linspace(x[0], x[-1], 500)
    y_fit = a + b * x_fit + c * x_fit ** 2
    y_spline_manual = evaluate_manual_spline(x_fit)
    y_spline_scipy = scipy_spline(x_fit)

    max_difference = np.max(np.abs(y_spline_manual - y_spline_scipy))
    node_error = np.max(np.abs(evaluate_manual_spline(x) - y))
    print(f"\nMaximum manual/SciPy difference: {max_difference:.3e}")
    print(f"Maximum interpolation error at nodes: {node_error:.3e}")

    plt.figure(figsize=(11, 6))
    plt.scatter(x, y, color="red", s=65, zorder=5, label="Source data")
    plt.plot(
        x_fit,
        y_fit,
        "--",
        color="blue",
        linewidth=2,
        label=f"Quadratic approximation: P(x) = {p_x}",
    )
    plt.plot(
        x_fit,
        y_spline_manual,
        color="green",
        linewidth=2,
        label="Natural cubic spline",
    )

    plt.title("Laboratory work 1, variant 4")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(linestyle=":", alpha=0.7)
    plt.tight_layout()
    plt.show()
