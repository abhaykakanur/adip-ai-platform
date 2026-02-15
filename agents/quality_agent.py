import os
import pandas as pd
import numpy as np


class QualityAgent:

    def __init__(self, processed_dir="data/processed", clean_dir="data/clean"):
        self.processed_dir = processed_dir
        self.clean_dir = clean_dir

        os.makedirs(self.clean_dir, exist_ok=True)

    def find_files(self):

        files = []

        for file in os.listdir(self.processed_dir):
            if file.endswith(".csv"):
                files.append(file)

        return files

    def load_data(self, filename):

        path = os.path.join(self.processed_dir, filename)
        df = pd.read_csv(path)

        return df

    def remove_duplicates(self, df):

        before = len(df)
        df = df.drop_duplicates()
        after = len(df)

        removed = before - after

        return df, removed

    def handle_missing(self, df):

        report = {}

        for col in df.columns:

            missing = df[col].isnull().sum()

            if missing > 0:

                if df[col].dtype in ["int64", "float64"]:
                    fill_value = df[col].median()
                    df[col] = df[col].fillna(fill_value)
                    strategy = "median"

                else:
                    df[col] = df[col].fillna("Unknown")
                    strategy = "constant:Unknown"

                report[col] = {
                    "missing": int(missing),
                    "strategy": strategy
                }

        return df, report

    def fix_dates(self, df):

        if "InvoiceDate" in df.columns:

            df["InvoiceDate"] = pd.to_datetime(
                df["InvoiceDate"],
                errors="coerce"
            )

        return df

    def clean_text(self, df):

        if "Description" in df.columns:

            df["Description"] = (
                df["Description"]
                .astype(str)
                .str.lower()
                .str.strip()
            )

        return df

    def detect_outliers(self, df):

        outlier_report = {}

        numeric_cols = df.select_dtypes(
            include=["int64", "float64"]
        ).columns

        for col in numeric_cols:

            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)

            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outliers = df[
                (df[col] < lower) | (df[col] > upper)
            ]

            outlier_report[col] = len(outliers)

        return outlier_report

    def save_clean(self, df, filename):

        out_path = os.path.join(self.clean_dir, filename)

        df.to_csv(out_path, index=False)

        return out_path

    def run(self):

        print("\n🧹 Quality Agent Started\n")

        files = self.find_files()

        if not files:
            print("❌ No processed files found")
            return

        for file in files:

            print(f"📄 Cleaning: {file}\n")

            df = self.load_data(file)

            df, removed = self.remove_duplicates(df)

            df, missing_report = self.handle_missing(df)

            df = self.fix_dates(df)

            df = self.clean_text(df)

            outliers = self.detect_outliers(df)

            output = self.save_clean(df, file)

            print("📉 Duplicates removed:", removed)

            print("\n🧪 Missing Values Handling:")
            for col, info in missing_report.items():
                print(f"   {col}: {info}")

            print("\n🚨 Outliers Detected:")
            for col, count in outliers.items():
                print(f"   {col}: {count}")

            print(f"\n💾 Clean data saved to: {output}\n")

        print("✅ Quality Check Complete\n")
