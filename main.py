from agents.ingestion_agent import IngestionAgent
from agents.quality_agent import QualityAgent
from agents.etl_agent import ETLAgent
from agents.analytics_agent import AnalyticsAgent
from agents.ml_agent import MLAgent
from agents.monitoring_agent import MonitoringAgent


def main():

    ingestion = IngestionAgent()
    ingestion.run()

    quality = QualityAgent()
    quality.run()

    etl = ETLAgent()
    etl.run()

    analytics = AnalyticsAgent()
    analytics.run()

    ml = MLAgent()
    ml.run()

    monitor = MonitoringAgent()
    monitor.run()


if __name__ == "__main__":
    main()
