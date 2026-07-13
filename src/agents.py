from typing import Annotated, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langchain_core.messages import ToolMessage
from src.tools import search_orders, search_invoices, get_order_status

class QueryType(BaseModel):
    query_type: Literal["invoice","order","general"] = Field(
        ...,
        description="Classify if the message was a query about invoices, orders or a general query. If the query is about invoices return invoice, else If the query is about orders return order, else return general"
    )

class State(TypedDict):
    messages: Annotated[list, add_messages]
    query_type: str | None

llm = ChatOllama(model="llama3.2", temperature=0.7)

orders_tools = [search_orders, get_order_status]
invoices_tools = [search_invoices]

def classify_query(state: State):
    message = state["messages"][-1]
     
    classifier_llm = llm.with_structured_output(QueryType)
    result = classifier_llm.invoke([
        {
            "role" : "system",
            "content" : """Classify the user message as either questions on invoices, orders or general.
            - 'order': If the user asks about an order or the query has the word order or orders in it. Or the number of orders or mentions order date or order details
            - 'invoice': If the user asks about an invoice or the query has the word invoice or invoices or mentions invoice date or invoice details
            - 'general' : If the user asks anything other than about invoices or orders """
        },
        {
            "role" : "user",
            "content" : message.content
        }
    ])
    print(result.query_type)
    return {"query_type": result.query_type}

def router(state: State):
    query_type = state.get("query_type" , "general")
    print(f"ROUTER {query_type}")
    if query_type == "order":
        return {"next" : "order"}
    elif query_type == "invoice":
        return {"next" : "invoice"}
    return {"next" : "general"}

def orders_agent(state: State):
    """Handle tool calls and responses properly"""
    llm_with_tools = llm.bind_tools(orders_tools)
    
    # Build proper message history
    messages = [
        {"role": "system", "content": "You are an assistant that helps with order queries. Use the available tools when needed to search for order information or check order status."}
    ] + state["messages"]
    
    response = llm_with_tools.invoke(messages)
    print(f"Orders agent response: {response}")
    
    # Check if the model wants to use tools
    if response.tool_calls:
        # Process tool calls
        tool_messages = []
        for tool_call in response.tool_calls:
            print(f"Tool call: {tool_call}")
            
            # Find and execute the tool
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            if tool_name == "search_orders":
                result = search_orders.invoke(tool_args.get('query'))
            elif tool_name == "get_order_status":
                result = get_order_status.invoke(tool_args.get('order_id'))
            else:
                result = f"Unknown tool: {tool_name}"
            
            # Create tool message
            tool_message = ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            )
            tool_messages.append(tool_message)
        
        # Get final response with tool results
        final_messages = messages + [response] + tool_messages
        final_response = llm.invoke(final_messages)
        
        return {"messages": [response] + tool_messages + [final_response]}
    else:
        # No tools needed
        return {"messages": [response]}

def general_agent(state: State):
    last_message = state["messages"][-1]

    messages = [
        { "role": "system",
          "content": """You are a business internal employees assistant focusing on helping them with orders, invoices or other
          common queries they may have."""
        },
        {
            "role":"user",
            "content":last_message.content
        }
    ]
    response = llm.invoke(messages)
    
    return {"messages": [response]}

def invoices_agent(state: State):
    """Handle tool calls and responses properly for invoices"""
    llm_with_tools = llm.bind_tools(invoices_tools)
    
    # Build proper message history
    messages = [
        {"role": "system", "content": "You are an assistant that helps with invoice queries. Use the available tools when needed to search for invoice information."}
    ] + state["messages"]
    
    response = llm_with_tools.invoke(messages)
    print(f"Invoice agent response: {response}")
    
    # Check if the model wants to use tools
    if response.tool_calls:
        # Process tool calls
        tool_messages = []
        for tool_call in response.tool_calls:
            print(f"Tool call: {tool_call}")
            
            # Find and execute the tool
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            if tool_name == "search_invoices":           
                result = search_invoices.invoke(tool_args.get('query'))
            else:
                result = f"Unknown tool: {tool_name}"
            
            # Create tool message
            tool_message = ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            )
            tool_messages.append(tool_message)
        
        # Get final response with tool results
        final_messages = messages + [response] + tool_messages
        final_response = llm.invoke(final_messages)
        
        return {"messages": [response] + tool_messages + [final_response]}
    else:
        # No tools needed
        return {"messages": [response]}
