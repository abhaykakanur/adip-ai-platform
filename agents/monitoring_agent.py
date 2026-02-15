import os
import pandas as pd
import numpy as np
import joblib


class MonitoringAgent:

    def __init__(self,
                 feature_dir="data/features",
                 model_dir="models",
                 monitor_dir="monitoring"):

        self.feature_dir = feature_dir
        self.model_dir = model_dir
        self.monitor_dir = monitor_dir

        os.makedirs(self.monitor_dir, exist_ok=True)

    def load_latest_data(self):

        files = [
            f for f in os.listdir(self.feature_dir)
            if f.endswith(".csv")
        ]

        if not files:
            raise ValueError("No feature files found")

        latest = max(files)
        path = os.path.join(self.feature_dir, latest)

        return pd.read_csv(path)

    def load_model(self):

        models = [
            f for f in os.listdir(self.model_dir)
            if f.endswith(".joblib")
        ]

        if not models:
            raise ValueError("No trained model found")

        best = models[0]
        path = os.path.join(self.model_dir, best)

        return joblib.load(path), best

    def detect_data_drift(self, df):

        drift_report = {}

        numeric_cols = df.select_dtypes(
            include=["int64", "float64"]
        ).columns

        for col in numeric_cols:

            drift_report[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std())
            }

        return drift_report

    def evaluate_model(self, model, df):

        if "Revenue" not in df.columns:
            return None

        X = df.drop(columns=["Revenue"])
        y = df["Revenue"]

        preds = model.predict(X)

        mse = np.mean((preds - y) ** 2)
        rmse = np.sqrt(mse)

        return {"rmse": float(rmse)}

    def save_report(self, drift, performance, model_name):

        path = os.path.join(
            self.monitor_dir,
            "monitor_report.txt"
        )

        with open(path, "w") as f:

            f.write("=== MODEL MONITORING REPORT ===\n\n")
            f.write(f"Model Used: {model_name}\n\n")

            f.write("DATA DISTRIBUTION\n")
            for col, stats in drift.items():
                f.write(f"{col}: {stats}\n")

            f.write("\nMODEL PERFORMANCE\n")
            f.write(str(performance))

        return path

    def trigger_retraining(self, rmse, threshold=1.0):

        if rmse > threshold:

            print("⚠️ Performance degraded")
            print("🔁 Triggering automatic retraining...\n")

            from agents.ml_agent import MLAgent

            ml = MLAgent()
            ml.run()

        else:
            print("✅ Model performance stable\n")

    def run(self):

        print("\n📡 Monitoring Agent Started\n")

        try:
            df = self.load_latest_data()
            model, name = self.load_model()

            drift = self.detect_data_drift(df)
            performance = self.evaluate_model(model, df)

            report = self.save_report(
                drift,
                performance,
                name
            )

            print("✅ Drift Analysis Complete")
            print("📊 Model Performance:", performance)
            print(f"📄 Report saved to: {report}\n")

            if performance and "rmse" in performance:
                self.trigger_retraining(
                    performance["rmse"]
                )

        except Exception as e:
            print(f"❌ Monitoring Error: {e}")

        print("✅ Monitoring Complete\n")
