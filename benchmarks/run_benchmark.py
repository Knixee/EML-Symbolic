import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from emlsymbolic import EMLSymbolicRegressor

def run_benchmark(X_train, X_test, y_train, y_test, params: dict) -> np.ndarray:
    
    if not params:
        raise ValueError("The 'params' dictionary is empty.")
    
    model = EMLSymbolicRegressor(
        **params,
        random_state=42,
        verbose=500
    )
    
    model.fit(X_train, y_train, eval_set=(X_test, y_test))
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"MSE: {mse:.6f}")
    print(f"R^2 Score: {r2:.4f}")
    print(f"Symbolic Equation (Raw): {model.to_symbolic()}")
    print(f"Standard Equation (Simplified): {model.to_standard_equation()}")
    
    return y_pred
