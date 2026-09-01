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
    pass

