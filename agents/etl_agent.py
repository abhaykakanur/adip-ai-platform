import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


class ETLAgent:

    def __init__(self, clean_dir="data/clean", feature_dir="data/features"):

        self.clean_dir = clean_dir
        self.feature_dir = feature_dir

        os.makedirs(self.feature_dir, exist_ok=True)

        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()

    def find_files(self):

        files = []

        for file in os.listdir(self.clean_dir):
            if file.endswith(".csv"):
                files.append(file)

        return files

    def load_data(self, filename):

        path = os.path.join(self.clean_dir, filename)
        df = pd.read_csv(path)

        return df

    def create_features(self, df):

        # Date features
        if "InvoiceDate" in df.columns:

            df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

            df["year"] = df["InvoiceDate"].dt.year
            df["month"] = df["InvoiceDate"].dt.month
            df["day"] = df["InvoiceDate"].dt.day
            df["weekday"] = df["InvoiceDate"].dt.weekday

        # Revenue feature
        if "Quantity" in df.columns and "UnitPrice" in df.columns:
            df["Revenue"] = df["Quantity"] * df["UnitPrice"]

        return df

    def encode_categorical(self, df):

        categorical_cols = df.select_dtypes(
            include=["object"]
        ).columns

        for col in categorical_cols:

            df[col] = df[col].astype(str)

            df[col] = self.encoder.fit_transform(df[col])

        return df

    def scale_numeric(self, df):

        numeric_cols = df.select_dtypes(
            include=["int64", "float64"]
        ).columns

        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])

        return df

    def select_features(self, df):

        drop_cols = []

        if "InvoiceDate" in df.columns:
            drop_cols.append("InvoiceDate")

        df = df.drop(columns=drop_cols)

        return df

    def save_features(self, df, filename):

        out_path = os.path.join(self.feature_dir, filename)

        df.to_csv(out_path, index=False)

        return out_path

    def run(self):

        print("\n⚙️ ETL Agent Started\n")

        files = self.find_files()

        if not files:
            print("❌ No clean files found")
            return

        for file in files:

            print(f"📄 Transforming: {file}\n")

            df = self.load_data(file)

            df = self.create_features(df)

            df = self.encode_categorical(df)

            df = self.scale_numeric(df)

            df = self.select_features(df)

            output = self.save_features(df, file)

            print("✅ Feature engineering complete")
            print(f"💾 Saved to: {output}\n")

        print("✅ ETL Pipeline Complete\n")
