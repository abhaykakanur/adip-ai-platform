import os
import pandas as pd


class IngestionAgent:

    def __init__(self, raw_dir="data/raw", processed_dir="data/processed"):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir

        os.makedirs(self.processed_dir, exist_ok=True)

    def find_files(self):
        files = []

        for file in os.listdir(self.raw_dir):
            if file.endswith(".csv") or file.endswith(".xlsx"):
                files.append(file)

        return files

    def load_file(self, filename):
        path = os.path.join(self.raw_dir, filename)

        if filename.endswith(".csv"):

            encodings = ["utf-8", "latin1", "ISO-8859-1", "cp1252"]

            for enc in encodings:
                try:
                    df = pd.read_csv(path, encoding=enc)
                    print(f"✅ Loaded with encoding: {enc}")
                    return df
                except UnicodeDecodeError:
                    print(f"⚠️ Failed with encoding: {enc}")
                    continue

            raise ValueError("❌ Could not decode CSV with known encodings")

        else:
            df = pd.read_excel(path)
            print("✅ Loaded Excel file")
            return df

    def analyze_schema(self, df):
        schema = {}

        for col in df.columns:
            schema[col] = str(df[col].dtype)

        return schema

    def profile_data(self, df):

        profile = {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicates": int(df.duplicated().sum())
        }

        return profile

    def save_processed(self, df, filename):

        out_path = os.path.join(self.processed_dir, filename)

        df.to_csv(out_path, index=False)

        return out_path

    def run(self):

        print("\n📥 Ingestion Agent Started\n")

        files = self.find_files()

        if not files:
            print("❌ No files found in raw directory")
            return

        for file in files:

            print(f"📄 Processing: {file}\n")

            try:
                df = self.load_file(file)

                schema = self.analyze_schema(df)
                profile = self.profile_data(df)

                output = self.save_processed(df, file)

                print("\n📊 Schema:")
                for k, v in schema.items():
                    print(f"   {k}: {v}")

                print("\n📈 Profile:")
                print(f"   Rows: {profile['rows']}")
                print(f"   Columns: {profile['columns']}")
                print(f"   Duplicates: {profile['duplicates']}")

                print(f"\n💾 Saved to: {output}\n")

            except Exception as e:
                print(f"❌ Error processing {file}: {e}")

        print("✅ Ingestion Complete\n")
