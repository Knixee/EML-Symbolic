import numpy as np
from sklearn.model_selection import train_test_split

from draw_benchmarks import draw_benchmark_results
from run_benchmark import run_benchmark

X = np.random.uniform(-1, 2, size=(2000, 1))
y = 1.0 / (1.0 + np.exp(-3.0 * (X[:, 0] - 0.5)))

params = {
    'n_layers': 4,
    'nodes_per_layer':3,
    'epochs': 5000,
    'split_ratio': 0.65,
    'l1_lambda': 0.005,
    'gate_l1_lambda': 0.0005,
    'pruning_threshold': 0.01,
    'lr_phase1': 0.02
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

print(f"\n==================================================")
print(f"  RUNNING BENCHMARK: 1 / (1 + exp(-3 * (x - 0.5)))  ")
print(f"==================================================")

y_pred = run_benchmark(X_train, X_test, y_train, y_test, params)

draw_benchmark_results(X_test, y_test, y_pred, title_name="1 / (1 + exp(-3 * (x - 0.5)))", bench_num=3)
