import argparse
import json
import logging
import os
import re

import torch
import transformers
from tqdm import tqdm

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

parser = argparse.ArgumentParser(description="Conversation Analysis")
parser.add_argument(
    "--model_id",
    type=str,
    default="meta-llama/Llama-3.2-1B-Instruct",
    help="Pretrained model ID",
)
parser.add_argument("--file_paths", nargs="+", help="Paths to conversation files")
parser.add_argument(
    "--output_dir", type=str, default="./", help="Directory to save results"
)

# Define analysis questions
QUESTIONS = [
    "Has any person given their age? (and what age was given)",
    "Has any person asked the other for their age?",
    "Has any person asked to meet up in person? Where?",
    "Has any person given a gift to the other? Or bought something from a list like an amazon wish list?",
    "Have any videos or photos been produced? Requested?",
]

# Updated prompt template without chat_history
PROMPT_TEMPLATE = """
    You are analyzing a conversation for specific patterns and behaviors. 
    Please review the following conversation carefully and answer each question precisely.

    Conversation:
    {conversation}

    For each question below, provide a clear YES/NO answer followed by specific evidence or "No evidence found":

    {questions}

    You are analyzing a conversation for specific patterns and behaviors. Review the conversation carefully and provide your analysis in the following the format:
    answer_question_1. NO - No evidence found in conversation
    answer_question_2. NO - No evidence found in conversation
    answer_question_3. Yes - xxxxxxx
    answer_question_4. NO - No evidence found in conversation
    answer_question_5. NO - No evidence found in conversation
    """


def setup_model(model_id):
    """Setup model and tokenizer."""
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token_id = tokenizer.eos_token_id

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto"
        )

        pipeline = transformers.pipeline(
            "text-generation", model=model, tokenizer=tokenizer, device_map="auto"
        )
        return pipeline
    except Exception as e:
        logging.error(f"Error setting up model: {e}")
        return None


def format_conversation(conversation_data):
    """Format conversation data into a readable string."""
    try:
        if isinstance(conversation_data, list):
            formatted_text = ""
            for conv in conversation_data:
                if "dialogue" in conv:
                    formatted_text += "Conversation:\n"
                    for turn in conv["dialogue"]:
                        formatted_text += f"{turn['speaker']}: {turn['text']}\n"
                    formatted_text += "\n"
            return formatted_text.strip()
        return str(conversation_data)
    except Exception as e:
        logging.error(f"Error formatting conversation: {e}")
        return str(conversation_data)


def load_conversation(file_path):
    """Load conversation from file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return format_conversation(data)
    except Exception as e:
        logging.error(f"Error loading file {file_path}: {e}")
        return None


def analyze_conversation(pipeline, conversation_text):
    """Analyze conversation using the model."""
    try:
        # Prepare questions string
        questions_str = "\n".join(
            f"[Question{i+1}]. {q}" for i, q in enumerate(QUESTIONS)
        )

        # Create prompt
        prompt = PROMPT_TEMPLATE.format(
            conversation=conversation_text, questions=questions_str
        )

        # Generate response
        response = pipeline(
            prompt, max_new_tokens=10000, do_sample=True, temperature=0.7, top_p=0.9
        )

        # Extract JSON from response
        response_text = response[0]["generated_text"]

        print(response_text)

        return None
    except Exception as e:
        logging.error(f"Error analyzing conversation: {e}")
        return None


def process_files(args):
    """Process all input files."""
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup model
    pipeline = setup_model(args.model_id)
    if not pipeline:
        return

    all_results = []

    # Process each file
    for file_path in tqdm(args.file_paths, desc="Processing files"):
        conversation_text = load_conversation(file_path)
        if not conversation_text:
            continue

        # Analyze conversation
        result = analyze_conversation(pipeline, conversation_text)
        if result:
            result["file_path"] = file_path
            all_results.append(result)

    # Save results
    output_path = os.path.join(args.output_dir, "analysis_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    logging.info(f"Analysis complete. Results saved to {output_path}")


if __name__ == "__main__":
    args = parser.parse_args()
    process_files(args)
