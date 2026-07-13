from langgraph.graph import StateGraph, START, END
from src.agents import State, classify_query, router, orders_agent, invoices_agent, general_agent

def create_graph():
    graph_builder = StateGraph(State)
    graph_builder.add_node("classifier", classify_query)
    graph_builder.add_node("router", router)
    graph_builder.add_node("order", orders_agent)
    graph_builder.add_node("invoice", invoices_agent)
    graph_builder.add_node("general", general_agent)

    graph_builder.add_edge(START, "classifier")
    graph_builder.add_edge("classifier", "router")
    graph_builder.add_conditional_edges(
        "router",
        lambda state: state.get("next"),
        {
            "order": "order",
            "invoice": "invoice",
            "general": "general"
        }
    )
    graph_builder.add_edge("order", END)
    graph_builder.add_edge("invoice", END)
    graph_builder.add_edge("general", END)

    return graph_builder.compile()

graph = create_graph()
