import os
import re
from typing import TypedDict, List
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import psycopg
import requests
from langgraph.types import interrupt
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")

# ---------------------- Define State ----------------------
class ClaimState(TypedDict):
    patient_id: str
    treatment_code: str
    claim_details: str
    patient_data: dict
    insurance_data: dict
    policy_docs: List[str]
    ai_validation_feedback: str
    final_decision: str
    _next: str  # ✅ Added _next for decision-making


# ---------------------- Constants ----------------------
FHIR_BASE_URL = "https://hapi.fhir.org/baseR4"

llm = ChatOpenAI(model="gpt-5", temperature=0.0)

# Load policy documents for RAG
loader = TextLoader("insurance_data.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(chunks, embeddings)


def hybrid_search(query: str, k: int = 4) -> List[Document]:
    query_terms = set(re.findall(r"\w+", query.lower()))
    semantic_results = vector_store.similarity_search_with_relevance_scores(query, k=k * 3)
    stored_documents = vector_store.get(include=["documents"]).get("documents", [])

    keyword_results = sorted(
        [
            (
                Document(page_content=document),
                len(query_terms & set(re.findall(r"\w+", document.lower())))
            )
            for document in stored_documents
        ],
        key=lambda item: item[1],
        reverse=True,
    )

    fused_scores = {}
    for rank, (document, _) in enumerate(semantic_results, start=1):
        fused_scores[document.page_content] = 1 / (60 + rank)
    for rank, (document, _) in enumerate(keyword_results, start=1):
        fused_scores[document.page_content] = fused_scores.get(document.page_content, 0) + 1 / (60 + rank)

    return [
        Document(page_content=content)
        for content, _ in sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)[:k]
    ]


# ---------------------- Step 1: Fetch Patient Data ----------------------
def fetch_patient_data(state: ClaimState):
    patient_id = state["patient_id"]
    response = requests.get(f"{FHIR_BASE_URL}/Patient/{patient_id}")
    if response.status_code == 200:
        state["patient_data"] = response.json()
    else:
        state["patient_data"] = {"error": f"Failed to fetch patient data for ID {patient_id}"}
    return state


# ---------------------- Step 2: Fetch Insurance Data ----------------------
def fetch_patient_insurance(state: ClaimState):
    patient_id = state["patient_id"]
    response = requests.get(f"{FHIR_BASE_URL}/Coverage?patient={patient_id}")
    if response.status_code == 200:
        state["insurance_data"] = response.json()
    else:
        state["insurance_data"] = {"error": f"Failed to fetch insurance data for patient ID {patient_id}"}
    return state

# ---------------------- Step 3: Retrieve Policy Documents ----------------------
def retrieve_policy_docs(state: ClaimState):
    treatment_code = state["treatment_code"]
    query = f"Retrieve policy details for treatment code {treatment_code}"
    docs = hybrid_search(query, k=4)
    state["policy_docs"] = [doc.page_content for doc in docs]
    return state

# ---------------------- Step 4: AI-Based Claim Validation ----------------------
def validate_claim(state: ClaimState):
    claim_text = f"""
    Claim Details: {state["claim_details"]}
    Patient Data: {state["patient_data"]}
    Insurance Data: {state["insurance_data"]}
    Policy Documents: {state["policy_docs"]}
    """

    response = llm.invoke(
        "Validate the following claim. Start your response with exactly one of: "
        "Decision: Approved, Decision: Rejected, or Decision: More Info. "
        f"Then explain the decision.\n{claim_text}"
    )
    state["ai_validation_feedback"] = response.content
    return state

# ---------------------- Step 5: Decision Node ----------------------
def claim_decision(state: ClaimState):
    decision_text = state["ai_validation_feedback"].strip()
    verdict_match = re.match(r"(?:decision\s*:\s*)?(approved|rejected|more info)\b", decision_text, re.IGNORECASE)
    verdict = verdict_match.group(1).lower() if verdict_match else ""

    if verdict == "more info":
        state["final_decision"] = "Request for more info"
        state["_next"] = "human_review"
    elif verdict == "approved":
        state["final_decision"] = "Approved"
        state["_next"] = "store_claim"
    else:
        state["final_decision"] = "Rejected"
        state["_next"] = "store_claim"
    return state


# ---------------------- Step 6: Store Decision in Database ----------------------
def store_claim(state: ClaimState):
    conn = psycopg.connect(DB_CONNECTION_STRING)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO claims (patient_id, status, decision_details) VALUES (%s, %s, %s)",
        (
            state["patient_id"],
            state["final_decision"],
            state["ai_validation_feedback"]
        ))
    
    conn.commit()
    cursor.close()
    conn.close()
    return state


# ---------------------- Step 7: Human Review ----------------------
def human_review(state: ClaimState):
   state["final_decision"] = interrupt(
       {
           "feedback": state["ai_validation_feedback"]
       }

    )
   return state

# ---------------------- Build LangGraph Workflow ----------------------


def create_workflow():
    graph = StateGraph(ClaimState)
    graph.add_node("fetch_patient_data", fetch_patient_data)
    graph.add_node("fetch_patient_insurance", fetch_patient_insurance)
    graph.add_node("retrieve_policy_docs", retrieve_policy_docs)
    graph.add_node("validate_claim", validate_claim)
    graph.add_node("claim_decision", claim_decision)
    graph.add_node("store_claim", store_claim)
    graph.add_node("human_review", human_review)

    # ✅ Define workflow transitions
    graph.set_entry_point("fetch_patient_data")
    graph.add_edge("fetch_patient_data", "fetch_patient_insurance")
    graph.add_edge("fetch_patient_insurance", "retrieve_policy_docs")
    graph.add_edge("retrieve_policy_docs", "validate_claim")
    graph.add_edge("validate_claim", "claim_decision")
    graph.add_edge("human_review", "store_claim")

    # ✅ Decision-making edges
    graph.add_conditional_edges(
        "claim_decision",
        lambda state: state["_next"],
        {
            "store_claim": "store_claim",
            "human_review": "human_review"
        }
    )

    # ✅ Create an InMemoryCheckpointer
    checkpointer = MemorySaver()
    graph = graph.compile(checkpointer=checkpointer)
    return graph