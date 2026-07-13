import logging
from langchain_core.messages import HumanMessage
from src.graph import graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run():
    state = {"messages": [], "query_type": None}
    
    print("Welcome to the LangGraph Chatbot! (Type 'exit' to quit)")
    while True:
        user_input = input("Message: ")
        if user_input.lower() == "exit":
            print("Bye..")
            break

        # Create proper message object
        user_message = HumanMessage(content=user_input)
        state["messages"] = state.get("messages", []) + [user_message] 
        
        # Invoke the graph
        try:
            result = graph.invoke(state)
            
            # Update state with results
            state = result
            
            # Print the last assistant message
            if result.get("messages") and len(result["messages"]) > 0:
                last_message = result["messages"][-1]
                if hasattr(last_message, 'content'):
                    print(f"Assistant: {last_message.content}")
                else:
                    print(f"Assistant: {last_message}")
        except Exception as e:
            logger.error(f"Error during graph execution: {e}")

if __name__ == "__main__":
    run()