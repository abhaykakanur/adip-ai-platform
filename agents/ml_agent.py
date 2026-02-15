import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


class MLAgent:

    def __init__(self,
                 feature_dir="data/features",
                 model_dir="models"):

        self.feature_dir = feature_dir
        self.model_dir = model_dir

        os.makedirs(self.model_dir, exist_ok=True)

    def find_files(self):

        return [
            f for f in os.listdir(self.feature_dir)
            if f.endswith(".csv")
        ]

    def load_data(self, filename):

        path = os.path.join(self.feature_dir, filename)
        return pd.read_csv(path)

    def prepare_data(self, df):

        if "Revenue" not in df.columns:
            raise ValueError("Revenue column not found in dataset")

        X = df.drop(columns=["Revenue"])
        y = df["Revenue"]

        return train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

    def train_models(self, X_train, y_train):

        models = {}

        # Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        models["LinearRegression"] = lr

        # Random Forest
        rf = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        models["RandomForest"] = rf

        return models

    def evaluate_models(self, models, X_test, y_test):

        results = {}

        for name, model in models.items():

            preds = model.predict(X_test)

            r2 = r2_score(y_test, preds)

            mse = mean_squared_error(y_test, preds)
            rmse = mse ** 0.5  # Manual RMSE (version-safe)

            results[name] = {
                "r2": r2,
                "rmse": rmse
            }

        return results

    def select_best(self, results):

        # Select model with highest R2
        best = max(
            results,
            key=lambda x: results[x]["r2"]
        )

        return best

    def save_model(self, model, name):

        path = os.path.join(
            self.model_dir,
            f"{name}.joblib"
        )

        joblib.dump(model, path)

        return path

    def run(self):

        print("\n🤖 ML Agent Started\n")

        files = self.find_files()

        if not files:
            print("❌ No feature files found")
            return

        for file in files:

            print(f"📄 Training on: {file}\n")

            try:
                df = self.load_data(file)

                X_train, X_test, y_train, y_test = \
                    self.prepare_data(df)

                models = self.train_models(
                    X_train,
                    y_train
                )

                results = self.evaluate_models(
                    models,
                    X_test,
                    y_test
                )

                print("📊 Evaluation Results:")
                for name, metrics in results.items():
                    print(
                        f"   {name}: "
                        f"R2={metrics['r2']:.4f}, "
                        f"RMSE={metrics['rmse']:.4f}"
                    )

                best = self.select_best(results)
                best_model = models[best]

                model_path = self.save_model(
                    best_model,
                    best
                )

                print(f"\n🏆 Best Model: {best}")
                print(f"💾 Saved to: {model_path}\n")

            except Exception as e:
                print(f"❌ Error during training: {e}")

        print("✅ ML Training Complete\n")
