import ollama
import json
import pandas as pd
from tqdm import tqdm
from .prompts1 import AGE_PROMPT, AGE_REQUEST_PROMPT, MEETUP_PROMPT, GIFT_PROMPT, MEDIA_PROMPT
import argparse

ALL_PROMPTS = {
    "Q1": AGE_PROMPT,
    "Q2": AGE_REQUEST_PROMPT,
    "Q3": MEETUP_PROMPT,
    "Q4": GIFT_PROMPT,
    "Q5": MEDIA_PROMPT
}

VALID_RESPONSES = {"YES", "NO"}

def format_conversation(conv):
    return '\n'.join([f"{t['speaker']}: {t['text']}" for t in conv['turns']])

def get_all_prompts(conversation):
    return {prompt_id: prompt.format(conversation=format_conversation(conversation)) for prompt_id, prompt in ALL_PROMPTS.items()}

def prompt_ollama(model, prompt):
    # response = ollama.generate(model, prompt)['response']
    # return response if response in VALID_RESPONSES else "NO"
    response = ollama.generate(model, prompt)['response']
    # Debugging the actual model output
    print(f"Raw response: {response}") 
    # Parse to get both answer and evidence
    if 'YES' in response.upper():
        if 'Evidence:' in response:
            evidence = response.split('Evidence:', 1)[1].strip()
            return "YES", evidence
        return "YES", "Evidence not specified"
    return "NO", "No evidence found in conversation"


def get_all_answers(conversation, model):
    # return {prompt_id: prompt_ollama(model, prompt) for prompt_id, prompt in get_all_prompts(conversation).items()}
    results = {}
    for prompt_id, prompt in get_all_prompts(conversation).items():
        answer, evidence = prompt_ollama(model, prompt)
        results[prompt_id] = {"answer": answer, "evidence": evidence}
    return results

def get_all_answers_for_conversations(conversations, model):
    return [{'id':conv['conversation_id'], **get_all_answers(conv, model)} for conv in tqdm(conversations)]


# parser = argparse.ArgumentParser()
# parser.add_argument('--input_file', type=str, required=True)
# parser.add_argument('--model', type=str, default='llama3.1')
# parser.add_argument('--output_file', type=str, required=True)
# args = parser.parse_args()

# input_file = args.input_file
# model = args.model
# output_file = args.output_file

# with open(input_file) as f:
#     data = json.load(f)

# answers = get_all_answers_for_conversations(data, model)
# df = pd.DataFrame(answers)
# df.to_csv(output_file, index=False)