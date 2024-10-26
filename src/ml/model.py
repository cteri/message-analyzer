import csv
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


class LlamaModel:
    def __init__(self, model_id="meta-llama/Llama-3.1-8B-Instruct"):
        # Set up model and tokenizer
        self.model_id = model_id
        self.pipeline = self.setup_model_and_tokenizer()

    def setup_model_and_tokenizer(self):
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
            tokenizer.pad_token_id = tokenizer.eos_token_id
            model = transformers.AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.float16,
                                                                      device_map="auto")
            return transformers.pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
        except Exception as e:
            logging.error(f"Error setting up model: {e}")
            return None

    def load_data(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif ext == '.txt' or ext == '.csv':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = [{"conversation_text": f.read().replace('\n', ' ')}]  # Read entire file and replace newlines
            else:
                logging.error(f"Unsupported file format: {ext}")
                return []
            return data
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            return []

    def ask_questions(self, conversation_data):
        results = []

        for convo in tqdm(conversation_data, desc="Asking questions"):
            conversation_text = convo.get('conversation_text', '')

            prompt = f"Here is a conversation:\n\n{conversation_text}\n\nNow answer the following questions based on this conversation:\n{questions}"

            try:
                output = self.pipeline(prompt, max_new_tokens=5000, do_sample=True, temperature=0.6, top_p=0.9)
                full_answer = output[0]['generated_text'].strip()
            except Exception as e:
                logging.error(f"Error generating answer: {e}")
                full_answer = "Error\n"

            results.append({
                "conversation_id": convo.get('id', 'unknown'),
                "answer": full_answer.strip()
            })

        return results

    def analysis(self, file_paths: list[str]) -> list[dict]:
        results = []
        for file_path in file_paths:
            data = self.load_data(file_path)
            if not data:
                logging.error(f"No data loaded for {file_path}.")
                continue
            analysis_results = self.ask_questions(data)
            results.append({"file_path": file_path, "result": analysis_results})
        return results
