from preprocess import preprocess_data, detect_target_column
from optimize import optimize_model
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

def automl_pipeline(csv_path, task_type="classification", target_col=None):
    X_train, X_test, y_train, y_test = preprocess_data(csv_path, target_col)
    best_model_name, best_model, best_params = optimize_model(X_train, X_test, y_train, y_test, task_type)

    # Score on test set
    y_pred = best_model.predict(X_test)

    print(f"\nBest Model: {best_model_name}")
    print("Best Hyperparameters:", best_params)

    if task_type == "classification":
        acc = accuracy_score(y_test, y_pred)
        print("Accuracy:", acc)
    else:
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)
        print("MSE:", mse)
        print("RMSE:", rmse)
        print("R² Score:", r2)

def predict_from_custom_csv(csv_path, target_col=None):
    model = joblib.load("best_model.pkl")
    df = pd.read_csv(csv_path)

    # Drop target column manually or detect it
    if target_col and target_col in df.columns:
        X = df.drop(columns=[target_col])
    else:
        inferred_target = detect_target_column(df)
        X = df.drop(columns=[inferred_target]) if inferred_target in df.columns else df

    # Preprocess
    X = pd.get_dummies(X)
    X = X.fillna(0)
    preds = model.predict(X)

    print("\nPredictions on Custom Dataset:\n", preds)

if __name__ == "__main__":
    # Example usage
    automl_pipeline("Housing.csv", task_type="regression", target_col="price")

    # To test predictions on custom input
    # predict_from_custom_csv("test_input.csv", target_col="Heart Disease")
