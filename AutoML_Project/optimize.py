import optuna
from models import get_model
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import joblib

def optimize_model(X_train, X_test, y_train, y_test, task_type="classification"):
    trial_results = []

    def objective(trial):
        if task_type == "classification":
            model_name = trial.suggest_categorical("model", [
                "xgboost", "lightgbm", "catboost",
                "logistic", "random_forest", "decision_tree", "svm", "naive_bayes"
            ])
        else:
            model_name = trial.suggest_categorical("model", [
                "xgboost", "lightgbm", "catboost",
                "linear", "ridge", "lasso", "random_forest", "decision_tree", "svr"
            ])

        # Hyperparameters
        params = {}
        if model_name in ["xgboost", "lightgbm", "catboost", "random_forest"]:
            params["n_estimators"] = trial.suggest_int("n_estimators", 50, 300)
        if model_name in ["xgboost", "lightgbm", "catboost"]:
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3)
        if model_name in ["xgboost", "lightgbm", "catboost", "random_forest", "decision_tree"]:
            params["max_depth"] = trial.suggest_int("max_depth", 3, 12)
        if model_name in ["ridge", "lasso"]:
            params["alpha"] = trial.suggest_float("alpha", 0.001, 10.0)
        if model_name in ["svm", "svr"]:
            params["C"] = trial.suggest_float("C", 0.1, 10.0)

        model = get_model(model_name, params, task_type)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        if task_type == "classification":
            acc = accuracy_score(y_test, preds)
            trial_results.append((model_name, params, acc))
            return acc
        else:
            r2 = r2_score(y_test, preds)
            trial_results.append((model_name, params, r2))
            return r2

    # Optimize
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    # Sort results by best metric
    trial_results.sort(key=lambda x: x[2], reverse=True)  # sort by accuracy or r²

    # Select best
    best_model_name, best_params, best_score = trial_results[0]
    best_model = get_model(best_model_name, best_params, task_type)
    best_model.fit(X_train, y_train)

    joblib.dump(best_model, "best_model.pkl")

    print(f"\nBest Model: {best_model_name}")
    print(f"Best Hyperparameters: {best_params}")
    if task_type == "classification":
        print(f"Accuracy: {best_score}")
    else:
        print(f"R² Score: {best_score}")

    return best_model_name, best_model, best_params


##################################################################################

# import optuna
# from models import get_model
# from sklearn.metrics import accuracy_score, mean_squared_error
# import joblib

# def optimize_model(X_train, X_test, y_train, y_test, task_type="classification"):
#     trial_results = []

#     def objective(trial):
#         # Choose model
#         if task_type == "classification":
#             model_name = trial.suggest_categorical("model", [
#                 "xgboost", "lightgbm", "catboost",
#                 "logistic", "random_forest", "decision_tree", "svm", "naive_bayes"
#             ])
#         else:
#             model_name = trial.suggest_categorical("model", [
#                 "xgboost", "lightgbm", "catboost",
#                 "linear", "ridge", "lasso", "random_forest", "decision_tree", "svr"
#             ])

#         # Set hyperparameters
#         params = {}
#         if model_name in ["xgboost", "lightgbm", "catboost", "random_forest"]:
#             params["n_estimators"] = trial.suggest_int("n_estimators", 50, 300)
#         if model_name in ["xgboost", "lightgbm", "catboost"]:
#             params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3)
#         if model_name in ["xgboost", "lightgbm", "catboost", "random_forest", "decision_tree"]:
#             params["max_depth"] = trial.suggest_int("max_depth", 3, 12)
#         if model_name in ["ridge", "lasso"]:
#             params["alpha"] = trial.suggest_float("alpha", 0.001, 10.0)
#         if model_name in ["svm", "svr"]:
#             params["C"] = trial.suggest_float("C", 0.1, 10.0)

#         # Train model
#         model = get_model(model_name, params, task_type)
#         model.fit(X_train, y_train)
#         preds = model.predict(X_test)

#         if task_type == "classification":
#             score = accuracy_score(y_test, preds)
#         else:
#             mse = mean_squared_error(y_test, preds)
#             score = -mse  # minimize MSE

#         # Store result
#         trial_results.append((model_name, params, -score if task_type == "regression" else score))
#         return score

#     # Run Optuna
#     direction = "maximize" if task_type == "classification" else "minimize"
#     study = optuna.create_study(direction=direction)
#     study.optimize(objective, n_trials=30)

#     # Print all results
#     print("\n--- All Model Results ---")
#     for i, (name, params, score) in enumerate(trial_results):
#         print(f"{i+1}. Model: {name}")
#         print(f"   Score: {'Accuracy' if task_type == 'classification' else 'MSE'} = {score}")
#         print(f"   Params: {params}\n")

#     # Ask user to choose
#     choice = int(input(f"Enter the number of the model you'd like to use (1-{len(trial_results)}): "))
#     chosen_name, chosen_params, chosen_score = trial_results[choice - 1]

#     final_model = get_model(chosen_name, chosen_params, task_type)
#     final_model.fit(X_train, y_train)
#     joblib.dump(final_model, "best_model.pkl")

#     return chosen_name, final_model, chosen_params
