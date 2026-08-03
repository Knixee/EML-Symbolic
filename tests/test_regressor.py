import numpy as np
import pytest
from emlsymbolic import EMLSymbolicRegressor

def test_fit_and_predict():
    X = np.linspace(-2, 2, 100).reshape(-1, 1)
    y = X[:, 0] ** 2

    model = EMLSymbolicRegressor(n_layers=1, nodes_per_layer=2, epochs=100, verbose=False)
    model.fit(X, y)
    
    preds = model.predict(X)
    assert preds.shape == y.shape
    assert not np.isnan(preds).any()

def test_symbolic_output():
    X = np.ones((20, 2))
    y = np.ones(20)

    model = EMLSymbolicRegressor(epochs=10, verbose=False)
    model.fit(X, y)

    eq = model.to_symbolic()
    assert isinstance(eq, str)
    assert len(eq) > 0