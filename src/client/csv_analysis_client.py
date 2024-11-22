import argparse
import json
import logging
import csv
import warnings
from pathlib import Path
from typing import List, Dict, Any
from src.ml.model import LlamaModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore')


class ConversationAnalyzer:
    def __init__(self, model_name: str):
        self.llama_model = LlamaModel(model_name)

    def analyze_conversation(self, conversation: Dict[str, Any]) -> Dict[str, Any]:
        try:
            results = self.llama_model.ask_questions([conversation])
            # Extract answers from the formatted response
            formatted_result = self.llama_model.clean_and_format_response(results, "")
            answers = {}
            for q in formatted_result["analysis"]["questions"]:
                question_num = f"Q{q['question_number']}"
                if q["answer"] == 'YES':
                    answers[question_num] = f"{q['answer']} - {q['evidence']}"
                else:
                    answers[question_num] = q["answer"]

            return {
                "id": conversation["conversation_id"],
                **answers
            }
        except Exception as e:
            logging.error(f"Error analyzing conversation {conversation['conversation_id']}: {e}")
            return {
                "id": conversation["conversation_id"],
                **{f"Q{i + 1}": "ERROR" for i in range(5)}
            }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", default="results.csv")
    parser.add_argument("--model", default="llama2")

    args = parser.parse_args()

    analyzer = ConversationAnalyzer(args.model)

    try:
        with open(args.input_file, 'r') as f:
            conversations = json.load(f)

        fieldnames = ['id'] + [f'Q{i + 1}' for i in range(5)]

        with open(args.output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for conv in conversations:
                result = analyzer.analyze_conversation(conv)
                writer.writerow(result)
                logging.info(f"Processed conversation {result['id']}")

    except Exception as e:
        logging.error(f"Error during processing: {e}")


if __name__ == "__main__":
    main()