from langchain.tools import tool
from src.vectorstore import orders_retriever, invoices_retriever

# RAG Tool for orders PDFs
@tool
def search_orders(query: str) -> str:
    """Search the orders PDFs for information on order details, shipping details, customer details, shipper details, products and number of orders """
    print(f"Inside search_orders {query}")
    docs = orders_retriever.invoke(query)
    print(docs)
    if not docs:
        return "No matches found."
    chunks = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        src = meta.get("source", "unknown")
        page = meta.get("page", None)
        where = f" (page {page})" if page is not None else ""
        chunks.append(f"[{i}] {src}{where}:\n{d.page_content[:800]}")
    return "\n\n".join(chunks)

# RAG tool for invoices PDFs
@tool
def search_invoices(query: str) -> str:
    """Search the invoices PDFs for information on invoice details"""
    docs = invoices_retriever.invoke(query)
    if not docs:
        return "No matches found."
    chunks = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        src = meta.get("source", "unknown")
        page = meta.get("page", None)
        where = f" (page {page})" if page is not None else ""
        chunks.append(f"[{i}] {src}{where}:\n{d.page_content[:800]}")
    return "\n\n".join(chunks)

# Fictional API call to get the order status
@tool("get_order_status", return_direct=False, description="Get the order status of an order id")
def get_order_status(order_id: str) -> str:
    statuses = ["PENDING", "PROCESSING", "SHIPPED", "DELIVERED", "CANCELLED"]
    idx = sum(ord(c) for c in order_id) % len(statuses)
    return f"Order {order_id} status: {statuses[idx]}"
