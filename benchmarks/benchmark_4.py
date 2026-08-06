import numpy as np
from sklearn.model_selection import train_test_split

from .draw_benchmarks import draw_benchmark_results
from .run_benchmark import run_benchmark

X = np.random.uniform(0.1, 2.5, size=(2000, 2))
y = (X[:, 0] ** 1.5) * np.log(X[:, 1] + 1.0)

params = {}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

print(f"\n==================================================")
print(f"RUNNING BENCHMARK: (x1^1.5) * log(x2 + 1.0)")
print(f"==================================================")

y_pred = run_benchmark(X_train, y_train, X_test, y_test, params)

draw_benchmark_results(X_test, y_test, y_pred, title_name="(x1^1.5) * log(x2 + 1.0)")
