import json

import ollama
import pandas as pd
from tqdm import tqdm

from .prompts import AGE_PROMPT as YES_NO_AGE_PROMPT
from .prompts import AGE_REQUEST_PROMPT as YES_NO_AGE_REQUEST_PROMPT
from .prompts import GIFT_PROMPT as YES_NO_GIFT_PROMPT
from .prompts import MEDIA_PROMPT as YES_NO_MEDIA_PROMPT
from .prompts import MEETUP_PROMPT as YES_NO_MEETUP_PROMPT
from .prompts1 import AGE_PROMPT as EVIDENCE_AGE_PROMPT
from .prompts1 import AGE_REQUEST_PROMPT as EVIDENCE_AGE_REQUEST_PROMPT
from .prompts1 import GIFT_PROMPT as EVIDENCE_GIFT_PROMPT
from .prompts1 import MEDIA_PROMPT as EVIDENCE_MEDIA_PROMPT
from .prompts1 import MEETUP_PROMPT as EVIDENCE_MEETUP_PROMPT

YES_NO_PROMPTS = {
    "Q1": YES_NO_AGE_PROMPT,
    "Q2": YES_NO_AGE_REQUEST_PROMPT,
    "Q3": YES_NO_MEETUP_PROMPT,
    "Q4": YES_NO_GIFT_PROMPT,
    "Q5": YES_NO_MEDIA_PROMPT,
}

EVIDENCE_PROMPTS = {
    "Q1": EVIDENCE_AGE_PROMPT,
    "Q2": EVIDENCE_AGE_REQUEST_PROMPT,
    "Q3": EVIDENCE_MEETUP_PROMPT,
    "Q4": EVIDENCE_GIFT_PROMPT,
    "Q5": EVIDENCE_MEDIA_PROMPT,
}


def format_conversation(conv):
    return "\n".join([f"{t['speaker']}: {t['text']}" for t in conv["turns"]])


def get_yes_no_answer(model, prompt):
    response = ollama.generate(model, prompt)["response"]
    # print(f"YES/NO Raw response: {response}")
    return "YES" if "YES" in response.upper() else "NO"


def get_evidence(model, prompt):
    response = ollama.generate(model, prompt)["response"]
    # print(f"Evidence Raw response: {response}")
    if "Evidence:" in response:
        return response.split("Evidence:", 1)[1].strip()
    return "Evidence not found"


def get_all_answers(conversation, model):
    formatted_conv = format_conversation(conversation)
    results = {}
    evidence_matches = {}

    for qid in YES_NO_PROMPTS.keys():
        # First get YES/NO
        yes_no_prompt = YES_NO_PROMPTS[qid].format(conversation=formatted_conv)
        answer = get_yes_no_answer(model, yes_no_prompt)

        # If YES, get evidence
        evidence = "No evidence found in conversation"
        if answer == "YES":
            evidence_prompt = EVIDENCE_PROMPTS[qid].format(conversation=formatted_conv)
            evidence = get_evidence(model, evidence_prompt)

            # Track individual evidence pieces
            if "and" in evidence:
                pieces = [p.strip() for p in evidence.split("and")]
            else:
                pieces = [evidence]

            evidence_matches[qid] = pieces  # Store for later matching

        results[qid] = {
            "answer": answer,
            "evidence": evidence,
            "evidence_pieces": evidence_matches.get(qid, []),
        }

    return results, evidence_matches


def get_all_prompts(conversation):
    formatted_conv = format_conversation(conversation)
    prompts = {}

    for qid, prompt_template in YES_NO_PROMPTS.items():
        prompts[qid] = prompt_template.format(conversation=formatted_conv)

    return prompts


def get_all_answers_for_conversations(conversations, model):
    results = []
    for conv in conversations:
        prompts = get_all_prompts(conv)
        answers = {
            prompt_id: get_yes_no_answer(model, prompt)
            for prompt_id, prompt in prompts.items()
        }
        results.append({"id": conv["conversation_id"], **answers})
    return results
