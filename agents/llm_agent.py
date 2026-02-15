import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class LLMInsightAgent:

    def __init__(self,
                 report_dir="data/reports",
                 insight_dir="data/insights"):

        self.report_dir = report_dir
        self.insight_dir = insight_dir

        os.makedirs(self.insight_dir, exist_ok=True)

        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )

    def find_latest_report(self):

        files = [
            f for f in os.listdir(self.report_dir)
            if f.endswith("_report.txt")
        ]

        if not files:
            raise ValueError("No reports found")

        return max(files)

    def read_report(self, filename):

        path = os.path.join(self.report_dir, filename)

        with open(path, "r") as f:
            return f.read()

    def generate_insight(self, report):

        prompt = f"""
You are a senior business data analyst.

Analyze the following report and give:

1. Key problems
2. Opportunities
3. Actionable recommendations
4. Risks

Report:
{report}
"""

        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a business analyst."},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content

    def save_insight(self, insight, filename):

        out_path = os.path.join(
            self.insight_dir,
            filename.replace("_report.txt", "_insight.txt")
        )

        with open(out_path, "w") as f:
            f.write(insight)

        return out_path

    def run(self):

        print("\n🧠 LLM Insight Agent Started\n")

        try:
            report_file = self.find_latest_report()
            report = self.read_report(report_file)

            insight = self.generate_insight(report)

            path = self.save_insight(insight, report_file)

            print("✅ AI Insights Generated")
            print(f"📄 Saved to: {path}\n")

        except Exception as e:
            print(f"❌ LLM Error: {e}")

        print("✅ LLM Agent Complete\n")
