import argparse
import os
import torch
import transformers
import json
import logging
from tqdm import tqdm

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

questions = """
1. Has any person given their age? (and what age was given)
2. Has any person asked the other for their age?
3. Has any person asked to meet up in person? Where?
4. Has any person given a gift to the other? Or bought something from a list like an amazon wish list?
5. Have any videos or photos been produced? Requested?
"""

parser = argparse.ArgumentParser(description="LLM question answering pipeline")
parser.add_argument('--model_id', type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Pretrained model ID")
parser.add_argument('--file_path', type=str, default="../../example.txt", help="Input file path")
args = parser.parse_args()


# Load data from a file (JSON or TXT)
def load_data(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = [{"conversation_text": f.read().replace('\n', ' ')}]  # Read entire file and replace newlines
        else:
            logging.error(f"Unsupported file format: {ext}")
            return []
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return []


# Set up model and tokenizer
def setup_model_and_tokenizer(model_id):
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        return transformers.pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    except Exception as e:
        logging.error(f"Error setting up model: {e}")
        return None


# Function to ask LLM the five questions without splitting conversation text
def ask_questions(pipeline, conversation_data):
    results = []

    for convo in tqdm(conversation_data, desc="Asking questions"):
        conversation_text = convo.get('conversation_text', '')

        prompt = f"Here is a conversation:\n\n{conversation_text}\n\nNow answer the following questions based on this conversation:\n{questions}"
        try:
            output = pipeline(prompt, max_new_tokens=5000, do_sample=True, temperature=0.6, top_p=0.9)
            full_answer = output[0]['generated_text'].strip()
        except Exception as e:
            logging.error(f"Error generating answer: {e}")
            full_answer = "Error\n"

        results.append({
            "conversation_id": convo.get('id', 'unknown'),
            "answer": full_answer.strip()
        })

    return results


# Main function to run pipeline and save the answers
def run_pipeline(file_path, model_id):
    data = load_data(file_path)
    if not data:
        logging.error("No data loaded.")
        return

    pipeline = setup_model_and_tokenizer(model_id)
    if pipeline is None:
        return

    results = ask_questions(pipeline, data)
    output_file = "llm_answers.json"

    # Save results to a file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    logging.info(f"Answers saved to {output_file}")


# Run pipeline and save results
if args.file_path:
    run_pipeline(args.file_path, args.model_id)