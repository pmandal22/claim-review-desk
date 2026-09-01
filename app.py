import json
import re

import chainlit as cl
from claim_processing_agent import create_workflow  # Replace with your actual graph module
from langgraph.types import Command

graph = create_workflow()

conversation_stage = "submit_claim"
claim_info = {}
thread = {"configurable": {"thread_id": "101"}}

CLAIM_ENTRY_PROMPT = """## Claim Review Desk

Review a treatment request against the patient's active coverage and supporting policy evidence. Claims are either decided automatically or routed to you when a manual decision is required.

### New claim

Send the three fields below in one message. You can also select a sample claim to begin.

```Claim
Patient ID: 137588944
Treatment Code: Z12.31
Claim Details: Screening mammogram requested as part of routine preventive care.
```
"""


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Screening mammogram",
            message=(
                "Patient ID: 137588944\n"
                "Treatment Code: Z12.31\n"
                "Claim Details: Screening mammogram requested as part of routine preventive care."
            ),
            icon="https://cdn.jsdelivr.net/npm/lucide-static@0.468.0/icons/clipboard-plus.svg",
        ),
        cl.Starter(
            label="Start a blank claim",
            message="Patient ID: \nTreatment Code: \nClaim Details: ",
            icon="https://cdn.jsdelivr.net/npm/lucide-static@0.468.0/icons/file-pen-line.svg",
        ),
    ]


@cl.on_chat_start
async def on_start():
    global conversation_stage, claim_info
    conversation_stage = "submit_claim"
    claim_info = {}
    await cl.Message(CLAIM_ENTRY_PROMPT).send()


@cl.on_message
async def handle_message(message):
    global conversation_stage, claim_info

    if conversation_stage == "submit_claim":
        claim_info, missing_fields = parse_claim_submission(message.content)
        if missing_fields:
            missing = ", ".join(missing_fields)
            await cl.Message(f"Please provide the following required fields: **{missing}**.\n\n{CLAIM_ENTRY_PROMPT}").send()
            return

        await cl.Message("Your claim is being reviewed. Please wait.").send()

        state = graph.invoke(claim_info, config=thread)
        # Human In The Loop
        tasks = graph.get_state(config=thread).tasks

        if tasks:
            feedback = tasks[0].interrupts[0].value.get("feedback")

            await cl.Message(f"## Manual Review Required\n\n{feedback}").send()

            await cl.Message(
                "Select a decision for this claim.",
                actions=[
                    cl.Action(name="approve_claim", payload={}, label="Approve", icon="check"),
                    cl.Action(name="reject_claim", payload={}, label="Reject", icon="x"),
                ],
            ).send()
            conversation_stage = "await_approval"
            return
        # No interrupt, display results directly
        await show_results(state)
        conversation_stage = "restart"
        return

    if conversation_stage == "await_approval":
        if message.content.strip().lower() == "yes":
            await complete_human_review("Approved")
        else:
            await complete_human_review("Rejected")
        return


    if conversation_stage == "restart" and message.content.strip().lower() == "restart":
        await start_new_claim()
        return

    await cl.Message("I could not process that response. Please follow the requested format.").send()


async def complete_human_review(decision):
    global conversation_stage
    state = graph.invoke(Command(resume=decision), config=thread)
    await show_results(state)
    await cl.Message("The claim decision has been recorded.").send()
    conversation_stage = "restart"


async def start_new_claim():
    global conversation_stage, claim_info
    claim_info = {}
    conversation_stage = "submit_claim"
    await cl.Message(CLAIM_ENTRY_PROMPT).send()


@cl.action_callback("approve_claim")
async def approve_claim(action):
    await action.remove()
    await complete_human_review("Approved")


@cl.action_callback("reject_claim")
async def reject_claim(action):
    await action.remove()
    await complete_human_review("Rejected")


@cl.action_callback("new_claim")
async def new_claim(action):
    await action.remove()
    await start_new_claim()


def parse_claim_submission(content):
    fields = {}
    field_names = {
        "patient id": "patient_id",
        "treatment code": "treatment_code",
        "claim details": "claim_details",
        "claim reason": "claim_details",
    }
    current_field = None

    for line in content.splitlines():
        match = re.match(r"^\s*([^:]+):\s*(.*)$", line)
        label = match.group(1).strip().lower() if match else ""
        if match and label in field_names:
            current_field = field_names[label]
            fields[current_field] = match.group(2).strip()
        elif current_field == "claim_details" and line.strip():
            fields[current_field] = f"{fields[current_field]} {line.strip()}".strip()

    missing_fields = [
        label
        for label, field_name in (
            ("Patient ID", "patient_id"),
            ("Treatment Code", "treatment_code"),
            ("Claim Details", "claim_details"),
        )
        if not fields.get(field_name)
    ]
    return fields, missing_fields


def format_policy_evidence(policy_documents):
    evidence_blocks = []

    for index, document in enumerate(policy_documents, start=1):
        try:
            policy = json.loads(document)
        except json.JSONDecodeError:
            evidence_blocks.append(f"**Evidence {index}**\n{document}")
            continue

        if not isinstance(policy, dict):
            evidence_blocks.append(f"**Evidence {index}**\n{str(policy)}")
            continue

        title = (
            policy.get("policy_name")
            or policy.get("procedure_name")
            or policy.get("procedure")
            or policy.get("definition")
            or "Policy details"
        )
        code = next(
            (
                policy.get(key)
                for key in ("procedure_code", "cpt_code", "icd_10_code")
                if policy.get(key)
            ),
            None,
        )
        coverage = policy.get("coverage") or policy.get("insurance_coverage")
        coverage_text = "; ".join(coverage) if isinstance(coverage, list) else coverage
        condition = (
            policy.get("policy_restrictions")
            or policy.get("exclusions")
            or policy.get("pre_authorization_required")
            or policy.get("pre_authorization")
        )
        condition_text = "; ".join(condition) if isinstance(condition, list) else condition

        details = [f"**{index}. {title}**" + (f" (`{code}`)" if code else "")]
        if coverage_text:
            details.append(f"Coverage: {coverage_text}")
        if condition_text:
            details.append(f"Condition: {condition_text}")
        details.append(f"```json\n{json.dumps(policy, indent=2)}\n```")
        evidence_blocks.append("\n".join(details))

    return "\n\n".join(evidence_blocks) or "No matching policy evidence was found."


def format_patient_summary(patient_data):
    if patient_data.get("error"):
        return patient_data["error"]

    name = patient_data.get("name", [{}])[0]
    full_name = " ".join(name.get("given", []) + [name.get("family", "")]).strip()
    return (
        f"- **Name:** {full_name or 'Not available'}\n"
        f"- **Patient ID:** {patient_data.get('id', 'Not available')}\n"
        f"- **Date of birth:** {patient_data.get('birthDate', 'Not available')}\n"
        f"- **Gender:** {patient_data.get('gender', 'Not available').title()}"
    )


def format_insurance_summary(insurance_data):
    if insurance_data.get("error"):
        return insurance_data["error"]

    entries = insurance_data.get("entry", [])
    coverage = entries[0].get("resource", {}) if entries else {}
    return (
        f"- **Plan:** {coverage.get('payor', [{}])[0].get('display', 'Not available')}\n"
        f"- **Status:** {coverage.get('status', 'Not available').title()}\n"
        f"- **Coverage ID:** {coverage.get('id', 'Not available')}"
    )


async def show_results(state):
    policy_evidence = format_policy_evidence(state.get("policy_docs", []))

    async with cl.Step(name="Policy Evidence", type="tool") as policy_step:
        policy_step.output = policy_evidence

    await cl.Message(
        f"""## Claim Summary

### Patient
{format_patient_summary(state["patient_data"])}

### Insurance
{format_insurance_summary(state["insurance_data"])}

### Assessment
{state['ai_validation_feedback']}

### Decision
**{state['final_decision']}**

Start a new review when you are ready.""",
        actions=[cl.Action(name="new_claim", payload={}, label="New claim", icon="plus")],
    ).send()