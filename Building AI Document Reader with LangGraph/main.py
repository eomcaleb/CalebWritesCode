import fitz
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph
from typing import TypedDict

class DocumentState(TypedDict):
    document_text: str
    document_summary: str

def document_extractor_agent(state: DocumentState) -> DocumentState:
    doc = fitz.open("document.pdf")
    document_text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return {
        **state,
        "document_text": document_text.strip()
    }

def document_summarizer_agent(state: DocumentState) -> DocumentState:
    api_key = "..."
    llm = ChatAnthropic(model='claude-3-5-sonnet-20240620', api_key=api_key)
    prompt = f"""Summarize the document in three sentences concisely: {state["document_text"]}"""
    summary = llm.invoke(prompt)
    return {
        **state,
        "document_summary": summary.content
    }
    
talking_documents = StateGraph(state_schema=DocumentState)
talking_documents.add_node("document_extractor", document_extractor_agent)
talking_documents.add_node("document_summarizer", document_summarizer_agent)
talking_documents.set_entry_point("document_extractor")
talking_documents.add_edge("document_extractor", "document_summarizer")
graph = talking_documents.compile()
state = {
    "document_text": "",
    "document_summary": ""
}
final_state = graph.invoke(state)
print(final_state["document_summary"])