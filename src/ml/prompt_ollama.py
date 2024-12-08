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
    return "YES" if "YES" in response.upper() else "NO"

def find_evidence_in_conversation(evidence_text, conversation_turns):
    """Find the actual conversation turn that contains the evidence."""
    matching_lines = []
    
    clean_evidence = evidence_text.lower()
    
    # Remove any speaker attribution pattern (Name: ) and quotes
    if ":" in clean_evidence:
        clean_evidence = clean_evidence.split(":", 1)[1]
    clean_evidence = clean_evidence.replace('"', '').replace("'", "").strip()
    
    # Look for this evidence in the actual conversation
    for i, turn in enumerate(conversation_turns):
        message = turn['text'].lower()
        if clean_evidence in message or message in clean_evidence:
            matching_lines.append(i)
    
    return matching_lines

def get_evidence(model, prompt, conversation_turns):
    response = ollama.generate(model, prompt)["response"]
    if "Evidence:" not in response:
        return "No evidence found in conversation", []
        
    evidence_text = response.split("Evidence:", 1)[1].strip()
    matching_line_indices = find_evidence_in_conversation(evidence_text, conversation_turns)
    
    if not matching_line_indices:
        return "No evidence found in conversation", []
    
    return evidence_text, matching_line_indices

def get_all_answers(conversation, model):
    formatted_conv = format_conversation(conversation)
    results = {}
    evidence_matches = {}

    for qid in YES_NO_PROMPTS.keys():
        # Get YES/NO
        yes_no_prompt = YES_NO_PROMPTS[qid].format(conversation=formatted_conv)
        answer = get_yes_no_answer(model, yes_no_prompt)

        # Get evidence and matching lines if YES
        evidence_text = "No evidence found in conversation"
        matching_lines = []
        if answer == "YES":
            evidence_prompt = EVIDENCE_PROMPTS[qid].format(conversation=formatted_conv)
            evidence_text, matching_lines = get_evidence(model, evidence_prompt, conversation["turns"])
            
            # If we found no matching lines but got a YES, change to NO
            if not matching_lines:
                answer = "NO"
                evidence_text = "No evidence found in conversation"

        results[qid] = {
            "answer": answer,
            "evidence": evidence_text,
            "evidence_lines": matching_lines
        }
        
        if matching_lines:
            evidence_matches[qid] = matching_lines

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
