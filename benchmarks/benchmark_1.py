import numpy as np
from sklearn.model_selection import train_test_split

from .draw_benchmarks import draw_benchmark_results
from .run_benchmark import run_benchmark

X = np.random.uniform(0, 3, size=(2000, 2))
y = np.exp(-0.5 * X[:, 0]) * np.cos(2 * np.pi * X[:, 1])

params = {}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

print(f"\n==================================================")
print(f"RUNNING BENCHMARK: exp(-0.5 * x1) * cos(2 * pi * x2)")
print(f"==================================================")

y_pred = run_benchmark(X_train, y_train, X_test, y_test, params)

draw_benchmark_results(X_test, y_test, y_pred, title_name="exp(-0.5 * x1) * cos(2 * pi * x2)")
