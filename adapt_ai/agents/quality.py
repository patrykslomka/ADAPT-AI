"""Quality Agent — hallucination detection and confidence scoring via Claude."""
from __future__ import annotations
import json
import logging
import re
import time
from typing import TYPE_CHECKING

from anthropic import Anthropic

from adapt_ai.agents.state import AgentState
from adapt_ai.config import settings
from adapt_ai.llmops.usage import record_llm_call

if TYPE_CHECKING:
    from adapt_ai.orchestrator.client import MCPClient

logger = logging.getLogger(__name__)

# Common valid clinical drug names — supplement the MCP validation tool.
# Ported from src/agents/quality_agent.py to catch hallucinated drug names before LLM.
_VALID_DRUGS: frozenset[str] = frozenset({
    "aspirin", "ibuprofen", "acetaminophen", "tylenol", "advil",
    "metformin", "lisinopril", "amlodipine", "metoprolol", "atorvastatin",
    "omeprazole", "losartan", "gabapentin", "sertraline", "levothyroxine",
    "hydrochlorothiazide", "furosemide", "prednisone", "azithromycin",
    "amoxicillin", "ciprofloxacin", "doxycycline", "ceftriaxone",
    "vancomycin", "piperacillin", "tazobactam", "meropenem",
    "heparin", "warfarin", "enoxaparin", "apixaban", "rivaroxaban",
    "insulin", "nitroglycerin", "morphine", "fentanyl", "ketamine",
    "propofol", "midazolam", "lorazepam", "diazepam", "haloperidol",
    "ondansetron", "diphenhydramine", "epinephrine", "norepinephrine",
    "dopamine", "dobutamine", "vasopressin", "phenylephrine",
    "methotrexate", "prednisone", "cyclosporine", "tacrolimus",
    "amiodarone", "digoxin", "spironolactone", "eplerenone",
    "clopidogrel", "ticagrelor", "prasugrel", "alteplase",
    "oseltamivir", "acyclovir", "valacyclovir", "remdesivir",
    "hydroxychloroquine", "chloroquine", "dexamethasone", "methylprednisolone",
    "sildenafil", "tadalafil", "finasteride", "tamsulosin",
    "albuterol", "salbutamol", "ipratropium", "tiotropium", "montelukast",
    "fluticasone", "budesonide", "beclomethasone",
    "lithium", "olanzapine", "quetiapine", "risperidone", "aripiprazole",
    "clozapine", "fluoxetine", "escitalopram", "citalopram", "paroxetine",
    "venlafaxine", "duloxetine", "bupropion", "mirtazapine",
    "carbamazepine", "valproate", "phenytoin", "levetiracetam", "lamotrigine",
    "rifampicin", "isoniazid", "pyrazinamide", "ethambutol",
    "trimethoprim", "sulfamethoxazole", "nitrofurantoin", "fosfomycin",
    "labetalol", "nicardipine", "hydralazine", "enalaprilat",
    "naloxone", "naltrexone", "buprenorphine", "methadone",
    "glucagon", "dextrose", "calcium", "potassium", "magnesium",
    "albumin", "mannitol", "furosemide", "acetazolamide",
    "paracetamol", "codeine", "tramadol", "oxycodone", "hydrocodone",
    "colchicine", "allopurinol", "febuxostat",
    "amlodipine", "nifedipine", "diltiazem", "verapamil",
    "captopril", "enalapril", "ramipril", "perindopril",
    "bisoprolol", "carvedilol", "atenolol", "propranolol",
    "simvastatin", "rosuvastatin", "pravastatin", "fluvastatin",
})

# Regex patterns for words that look like drug names (common pharmaceutical suffixes).
_DRUG_SUFFIX_RE = re.compile(
    r"\b[A-Z][a-z]+(?:mycin|cillin|prazole|olol|pine|statin|pril|sartan|mab|nib|vir|tide|zole|oxin)\b"
)

# Words that match drug suffixes but are not drugs.
_DRUG_FALSE_POSITIVES: frozenset[str] = frozenset({
    "medicine", "routine", "baseline", "pipeline", "outline", "guideline",
    "timeline", "crystalline", "membrane", "cocaine", "codeine",
})


def _check_drug_names(content: str) -> list[str]:
    """Return list of warning strings for potential hallucinated drug names.

    Finds words with pharmaceutical suffixes not in the known drug list.
    Two-stage: regex heuristic first (cheap), then validated against _VALID_DRUGS.
    """
    potential = {m.lower() for m in _DRUG_SUFFIX_RE.findall(content)}
    unknown = potential - _VALID_DRUGS - _DRUG_FALSE_POSITIVES
    return [f'Unrecognised drug name "{d}" — verify accuracy' for d in sorted(unknown)]


_QUALITY_SYSTEM = """\
You are a medical quality assurance specialist. Evaluate a clinical response for accuracy.

Given the original question and the AI response, assess:
1. Does the response directly address the question?
2. Is the clinical reasoning sound and consistent with medical evidence?
3. If an answer choice (A/B/C/D/E) is stated, is it plausible given the clinical picture?
4. Are there any hallucinations, contradictions, or unsupported claims?

Respond ONLY with a JSON object in this exact format:
{
  "passed": true or false,
  "score": 0.0-1.0,
  "issues": ["issue1", "issue2"],
  "feedback": "brief corrective feedback if failed, else empty string"
}

Score ≥ 0.85 → passed=true. Be strict: fail if the response contains factual errors, omits \
critical safety information, contradicts established clinical guidelines, makes unsupported \
claims, or has any issue rated "Major" or "Critical". Scores of 0.6–0.84 should be \
passed=false with corrective feedback.\
"""


def make_quality_node(mcp_client: "MCPClient", anthropic_client: Anthropic):
    """Return a LangGraph node function for the Quality Agent."""

    async def quality_agent(state: AgentState) -> dict:
        query = state["query"]
        primary_response = state.get("primary_response", "")
        context = state.get("retrieved_context", "")

        if not primary_response:
            statuses = {**state.get("agent_statuses", {}), "quality": "skipped"}
            return {
                "quality_result": {"passed": True, "score": 0.5, "issues": [], "feedback": ""},
                "agent_statuses": statuses,
            }

        # Pre-LLM drug name check (cheap regex, catches hallucinated drug names).
        drug_warnings = _check_drug_names(primary_response)

        evaluation_prompt = (
            f"Original question:\n{query}\n\n"
            f"Clinical context used:\n{context[:800] if context else 'None'}\n\n"
        )
        if drug_warnings:
            evaluation_prompt += (
                f"[Pre-check flags — verify these in the response below:]\n"
                + "\n".join(f"  • {w}" for w in drug_warnings)
                + "\n\n"
            )
        evaluation_prompt += f"AI response to evaluate:\n{primary_response}"

        try:
            t0 = time.perf_counter()
            resp = anthropic_client.messages.create(
                model=settings.model_name,
                max_tokens=512,
                temperature=0.0,
                system=_QUALITY_SYSTEM,
                messages=[{"role": "user", "content": evaluation_prompt}],
            )
            record_llm_call(
                agent="quality",
                model=settings.model_name,
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
                latency_s=time.perf_counter() - t0,
                run_id=state["session_id"],
            )
            raw = resp.content[0].text.strip()
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                result = json.loads(m.group(0))
            else:
                logger.warning("Quality agent returned non-JSON: %s", raw[:200])
                result = {"passed": False, "score": 0.0, "issues": ["Quality agent returned unparseable output"], "feedback": "Re-evaluate and provide a well-structured clinical response."}
        except Exception as e:
            logger.warning("Quality agent error: %s", e)
            result = {"passed": False, "score": 0.0, "issues": [f"Quality evaluation failed: {e}"], "feedback": "Re-evaluate and provide a well-structured clinical response."}

        status = "approved" if result.get("passed") else "rejected"
        statuses = {**state.get("agent_statuses", {}), "quality": status}
        return {
            "quality_result": result,
            "revision_feedback": result.get("feedback", ""),
            "agent_statuses": statuses,
        }

    return quality_agent
