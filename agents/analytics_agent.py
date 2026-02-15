import os
import pandas as pd
import matplotlib.pyplot as plt


class AnalyticsAgent:

    def __init__(self, feature_dir="data/features", report_dir="data/reports"):

        self.feature_dir = feature_dir
        self.report_dir = report_dir

        os.makedirs(self.report_dir, exist_ok=True)

    def find_files(self):

        files = []

        for file in os.listdir(self.feature_dir):
            if file.endswith(".csv"):
                files.append(file)

        return files

    def load_data(self, filename):

        path = os.path.join(self.feature_dir, filename)
        df = pd.read_csv(path)

        return df

    def compute_kpis(self, df):

        kpis = {}

        if "Revenue" in df.columns:
            kpis["total_revenue"] = df["Revenue"].sum()
            kpis["avg_revenue"] = df["Revenue"].mean()

        if "Quantity" in df.columns:
            kpis["total_quantity"] = df["Quantity"].sum()
            kpis["avg_quantity"] = df["Quantity"].mean()

        kpis["total_orders"] = len(df)

        return kpis

    def detect_anomalies(self, df):

        anomalies = {}

        if "Revenue" in df.columns:

            mean = df["Revenue"].mean()
            std = df["Revenue"].std()

            threshold = mean + 3 * std

            outliers = df[df["Revenue"] > threshold]

            anomalies["high_revenue_orders"] = len(outliers)

        return anomalies

    def plot_revenue_trend(self, df, filename):

        if "month" not in df.columns or "Revenue" not in df.columns:
            return None

        trend = df.groupby("month")["Revenue"].sum()

        plt.figure()
        trend.plot(title="Monthly Revenue Trend")
        plt.xlabel("Month")
        plt.ylabel("Revenue")

        out_path = os.path.join(
            self.report_dir,
            filename.replace(".csv", "_trend.png")
        )

        plt.savefig(out_path)
        plt.close()

        return out_path

    def save_report(self, kpis, anomalies, charts, filename):

        report_path = os.path.join(
            self.report_dir,
            filename.replace(".csv", "_report.txt")
        )

        with open(report_path, "w") as f:

            f.write("=== BUSINESS ANALYTICS REPORT ===\n\n")

            f.write("KEY PERFORMANCE INDICATORS\n")
            for k, v in kpis.items():
                f.write(f"{k}: {v}\n")

            f.write("\nANOMALIES\n")
            for k, v in anomalies.items():
                f.write(f"{k}: {v}\n")

            f.write("\nCHARTS\n")
            for chart in charts:
                if chart:
                    f.write(chart + "\n")

        return report_path

    def run(self):

        print("\n📊 Analytics Agent Started\n")

        files = self.find_files()

        if not files:
            print("❌ No feature files found")
            return

        for file in files:

            print(f"📄 Analyzing: {file}\n")

            df = self.load_data(file)

            kpis = self.compute_kpis(df)

            anomalies = self.detect_anomalies(df)

            charts = []

            chart_path = self.plot_revenue_trend(df, file)
            if chart_path:
                charts.append(chart_path)

            report = self.save_report(
                kpis,
                anomalies,
                charts,
                file
            )

            print("✅ KPIs Generated")
            print("🚨 Anomalies Detected:", anomalies)
            print("📈 Charts Created")
            print(f"📄 Report saved to: {report}\n")

        print("✅ Analytics Complete\n")
