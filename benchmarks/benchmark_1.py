import numpy as np
from sklearn.model_selection import train_test_split

from draw_benchmarks import draw_benchmark_results
from run_benchmark import run_benchmark

X = np.random.uniform(0, 5, size=(2000, 1))
y = np.sin(X[:, 0]) * np.exp(-0.1 * X[:, 0]) + 0.5 * (X[:, 0]**2)

params = {
    'n_layers': 3,
    'nodes_per_layer': 3,
    'epochs': 15000,
    'split_ratio': 0.75,
    'lr_phase1': 0.01,
    'lr_phase2': 0.001,
    'l1_lambda': 0.05,
    'pruning_threshold': 0.03,
    'clamp_min': -5.0,
    'clamp_max': 5.0,
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

print(f"\n==================================================")
print(f"RUNNING BENCHMARK: sin(x) * exp(-0.1 * x) + 0.5 * (x^2)")
print(f"==================================================")

y_pred = run_benchmark(X_train, X_test, y_train, y_test, params)

draw_benchmark_results(X_test, y_test, y_pred, title_name="sin(x) * exp(-0.1 * x) + 0.5 * (x^2)", bench_num=1)
