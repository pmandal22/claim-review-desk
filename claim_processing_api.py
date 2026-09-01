from fastapi import FastAPI
from pydantic import BaseModel
from claim_processing_agent import create_workflow

app = FastAPI()
graph = create_workflow()


class ClaimRequest(BaseModel):
    patient_id: str
    treatment_code: str
    claim_details: str


class ClaimResponse(BaseModel):
    final_decision: str
    ai_feedback: str


@app.post("/process-claim", response_model=ClaimResponse)
async def process_claim(request: ClaimRequest):
    input_state = {
        "patient_id": request.patient_id,
        "treatment_code": request.treatment_code,
        "claim_details": request.claim_details,
    }

    result = graph.invoke(
        input_state, config={"configurable": {"thread_id": "api-thread"}}
    )
    return {
        "final_decision": result.get("final_decision"),
        "ai_feedback": result.get("ai_validation_feedback"),
    }