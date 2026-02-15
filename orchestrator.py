from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END

from agents.ingestion_agent import IngestionAgent
from agents.quality_agent import QualityAgent
from agents.etl_agent import ETLAgent
from agents.analytics_agent import AnalyticsAgent
from agents.ml_agent import MLAgent
from agents.monitoring_agent import MonitoringAgent
from agents.llm_agent import LLMInsightAgent


# ---------------- CONFIG ---------------- #

MAX_RETRIES = 2


# ---------------- STATE ---------------- #

class PipelineState(TypedDict):
    step: str
    error: Optional[str]
    retries: int


# ---------------- AGENT NODES ---------------- #

def router_node(state: PipelineState):
    return state


def ingestion_node(state: PipelineState):

    try:
        IngestionAgent().run()
        return {
            "step": "ingestion_done",
            "error": None,
            "retries": state.get("retries", 0)
        }

    except Exception as e:
        return {
            "step": "ingestion_failed",
            "error": str(e),
            "retries": state.get("retries", 0)
        }


def quality_node(state: PipelineState):

    try:
        QualityAgent().run()
        return {
            "step": "quality_done",
            "error": None,
            "retries": state.get("retries", 0)
        }

    except Exception as e:
        return {
            "step": "quality_failed",
            "error": str(e),
            "retries": state.get("retries", 0)
        }


def etl_node(state: PipelineState):

    try:
        ETLAgent().run()
        return {
            "step": "etl_done",
            "error": None,
            "retries": state.get("retries", 0)
        }

    except Exception as e:
        return {
            "step": "etl_failed",
            "error": str(e),
            "retries": state.get("retries", 0)
        }


def analytics_node(state: PipelineState):

    try:
        AnalyticsAgent().run()
        return {
            "step": "analytics_done",
            "error": None,
            "retries": state.get("retries", 0)
        }

    except Exception as e:
        return {
            "step": "analytics_failed",
            "error": str(e),
            "retries": state.get("retries", 0)
        }


def ml_node(state: PipelineState):

    try:
        MLAgent().run()
        return {
            "step": "ml_done",
            "error": None,
            "retries": state.get("retries", 0)
        }

    except Exception as e:
        return {
            "step": "ml_failed",
            "error": str(e),
            "retries": state.get("retries", 0)
        }


def monitor_node(state: PipelineState):

    try:
        MonitoringAgent().run()
        return {
            "step": "monitor_done",
            "error": None,
            "retries": state.get("retries", 0)
        }

    except Exception as e:
        return {
            "step": "monitor_failed",
            "error": str(e),
            "retries": state.get("retries", 0)
        }


def llm_node(state: PipelineState):

    try:
        LLMInsightAgent().run()
        return {
            "step": "llm_done",
            "error": None,
            "retries": state.get("retries", 0)
        }

    except Exception as e:

        # LLM is optional → do NOT crash pipeline
        print("⚠️ LLM failed. Skipping insights.")

        return {
            "step": "llm_done",
            "error": None,
            "retries": state.get("retries", 0)
        }


# ---------------- ROUTER ---------------- #

def route(state: PipelineState):

    retries = state.get("retries", 0)

    # ---------- Error Handling ---------- #

    if state["error"] is not None:

        if retries < MAX_RETRIES:

            print(
                f"⚠️ Error detected. Retrying "
                f"({retries + 1}/{MAX_RETRIES})..."
            )

            return {
                "step": "ingestion",
                "error": None,
                "retries": retries + 1
            }

        else:

            print("\n❌ Max retries reached. Stopping pipeline.")
            print("Last error:", state["error"])

            return END

    # ---------- Normal Flow ---------- #

    step = state["step"]

    if step == "start":
        return "ingestion"

    if step == "ingestion_done":
        return "quality"

    if step == "quality_done":
        return "etl"

    if step == "etl_done":
        return "analytics"

    if step == "analytics_done":
        return "ml"

    if step == "ml_done":
        return "monitor"

    if step == "monitor_done":
        return "llm"

    if step == "llm_done":
        return END

    return END


# ---------------- GRAPH ---------------- #

def build_graph():

    graph = StateGraph(PipelineState)

    # Register nodes
    graph.add_node("router", router_node)

    graph.add_node("ingestion", ingestion_node)
    graph.add_node("quality", quality_node)
    graph.add_node("etl", etl_node)
    graph.add_node("analytics", analytics_node)
    graph.add_node("ml", ml_node)
    graph.add_node("monitor", monitor_node)
    graph.add_node("llm", llm_node)

    # Routing
    graph.add_conditional_edges("router", route)

    # Return to router
    graph.add_edge("ingestion", "router")
    graph.add_edge("quality", "router")
    graph.add_edge("etl", "router")
    graph.add_edge("analytics", "router")
    graph.add_edge("ml", "router")
    graph.add_edge("monitor", "router")
    graph.add_edge("llm", "router")

    # Entry
    graph.set_entry_point("router")

    return graph.compile()


# ---------------- RUN ---------------- #

if __name__ == "__main__":

    print("\n🧠 Autonomous Data Intelligence Platform Started\n")

    app = build_graph()

    final_state = app.invoke(
        {
            "step": "start",
            "error": None,
            "retries": 0
        }
    )

    print("\n✅ System Finished")
    print("Final State:", final_state)
