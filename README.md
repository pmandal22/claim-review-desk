# Claim Review Desk

An AI-assisted health insurance claim review workflow. A [LangGraph](https://langchain-ai.github.io/langgraph/) agent pulls patient and coverage records from a FHIR server, retrieves the relevant policy language via hybrid RAG, asks an LLM to adjudicate the claim, and either records the decision in Postgres or pauses for a human reviewer. A [Chainlit](https://docs.chainlit.io) chat UI drives the whole thing.

## How it works

```
fetch_patient_data → fetch_patient_insurance → retrieve_policy_docs → validate_claim → claim_decision
                                                                                            │
                                                              Approved / Rejected ──────────┤
                                                                                            │
                                                                    More Info → human_review (interrupt)
                                                                                            │
                                                                                       store_claim
```

| Node | What it does |
| --- | --- |
| `fetch_patient_data` | `GET /Patient/{id}` against the public HAPI FHIR R4 server |
| `fetch_patient_insurance` | `GET /Coverage?patient={id}` against the same server |
| `retrieve_policy_docs` | Hybrid search (semantic + keyword, fused with reciprocal rank fusion) over the policy corpus in Chroma |
| `validate_claim` | Asks the LLM for a verdict prefixed with `Decision: Approved` / `Rejected` / `More Info`, plus a rationale |
| `claim_decision` | Parses the verdict and routes: a clear approval/rejection goes straight to storage, `More Info` goes to a human |
| `human_review` | Calls LangGraph's `interrupt()` so the graph suspends; the UI surfaces Approve / Reject buttons and resumes with the reviewer's choice |
| `store_claim` | Inserts `(patient_id, status, decision_details)` into the `claims` table |

Graph state is checkpointed with an in-memory `MemorySaver`, which is what makes the interrupt-and-resume round trip possible. Because it is in-memory, state is lost when the process restarts.

## Layout

- [app.py](app.py) — Chainlit UI: claim intake parsing, the human-review action buttons, and the formatters that render patient, coverage, and policy evidence as Markdown
- [claim_processing_agent.py](claim_processing_agent.py) — the LangGraph workflow, hybrid retriever, and all node implementations
- [insurance_data.txt](insurance_data.txt) — the policy corpus (JSON records) that gets chunked and embedded at import time
- [claims.sql](claims.sql) — schema for the `claims` table
- [test data.txt](test%20data.txt) — three sample claims, one per policy path (clean approval, flagged facility, needs justification)
- [.chainlit/config.toml](.chainlit/config.toml), [public/minimal.css](public/minimal.css) — UI configuration and styling
- [claim_processing_api.py](claim_processing_api.py) — FastAPI wrapper exposing the `/process-claim` endpoint
- [main.py](main.py) — leftover IDE scaffolding, not part of the app

## Setup

Requires Python 3.12+, a reachable PostgreSQL instance, and an OpenAI API key. Run all commands from the repository root: the agent loads `insurance_data.txt` through a relative path when it starts.

```bash
uv sync
# Or: pip install -r requirements.txt
```

The current dependency manifests omit three direct runtime imports. Install them after the command above:

```bash
uv pip install python-dotenv requests langchain-text-splitters
# Or: pip install python-dotenv requests langchain-text-splitters
```

Create the claims table:

```bash
psql "$DB_CONNECTION_STRING" -f claims.sql
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
DB_CONNECTION_STRING=postgresql://user:password@localhost:5432/claims
```

`OPENAI_API_KEY` powers both the `gpt-5` adjudication call and the OpenAI embeddings used for retrieval.

> The FHIR endpoint is a public sandbox. Use only its sample identifiers; do not submit real patient data or other protected health information.

## Running

### Chainlit UI

```bash
chainlit run app.py -w
```

The app opens with a submission template. Paste a claim as a single message:

```
Patient ID: 137588944
Treatment Code: Z12.31
Claim Details: Routine screening colonoscopy performed on patient aged 50, as part of preventive care.
```

The parser accepts `Patient ID`, `Treatment Code`, and `Claim Details` (or `Claim Reason`) labels case-insensitively, and folds unlabeled continuation lines into the claim details. Missing fields are re-prompted before the graph runs.

If the LLM returns `More Info`, the run halts and the UI shows the model's rationale alongside Approve / Reject buttons. Your choice becomes the recorded `final_decision`. Otherwise the decision is stored immediately and the summary — patient, coverage, assessment, decision, and the retrieved policy evidence in a collapsible step — is rendered directly.

Try all three claims in [test data.txt](test%20data.txt) to exercise each branch.

### FastAPI endpoint

Start the API server with:

```bash
uvicorn claim_processing_api:app --reload
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

Submit a claim with `POST /process-claim`:

```bash
curl -X POST http://127.0.0.1:8000/process-claim \
     -H "Content-Type: application/json" \
     -d '{
          "patient_id": "137588944",
          "treatment_code": "Z12.31",
          "claim_details": "Screening mammogram requested as part of routine preventive care."
     }'
```

Successful responses contain the final decision and the model's assessment:

```json
{
     "final_decision": "Approved",
     "ai_feedback": "Decision: Approved ..."
}
```

When the model requests more information, `final_decision` is `"Request for more info"`. The API currently does not expose an endpoint to resume that interrupted human-review workflow; use the Chainlit UI to approve or reject those claims. The API also uses one fixed checkpoint thread (`"api-thread"`), so it is a demonstration endpoint rather than a concurrent multi-user service.

## Known limitations

- **Single session.** `conversation_stage`, `claim_info`, and the thread id (`"101"`) are module-level globals in [app.py](app.py), so concurrent users share one conversation and one checkpoint thread.
- **Ephemeral checkpoints.** `MemorySaver` keeps graph state in process memory; `langgraph-checkpoint-postgres` is already a dependency if you want to swap it in.
- **Startup cost.** The policy corpus is re-embedded into a fresh in-memory Chroma collection on every import of `claim_processing_agent`.
- **No test suite.**
- **Public FHIR server.** Patient lookups hit `hapi.fhir.org`, a shared sandbox; do not submit real patient identifiers or protected health information.
- **API checkpoint reuse.** Every API request uses the same LangGraph thread id (`"api-thread"`), which can mix checkpoint state between requests.
