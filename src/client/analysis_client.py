import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
warnings.filterwarnings('ignore')


class ConversationAnalyzer:
    def __init__(self, model_names: List[str]):
        self.model_names = model_names
        self.questions = {
            "Q1": "Does the conversation mention violence?",
            "Q2": "Does the conversation contain explicit content?",
            "Q3": "Is there a discussion about or planning of a date/meeting?",
            "Q4": "Are there signs of emotional distress?",
            "Q5": "Is there evidence of conflict?"
        }

    def analyze_conversation(self, conversation: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single conversation with all models."""
        results = {
            "original_id": conversation["conversation_id"],
            "dialogue": conversation["turns"],
            "model_analysis": {}
        }

        # Save partial results after each model's analysis
        for model_name in self.model_names:
            try:
                # Here you would implement the actual model analysis
                # For now, we'll just create placeholder results
                model_results = {
                    f"Q{i}": "Not analyzed yet"
                    for i in range(1, len(self.questions) + 1)
                }
                results["model_analysis"][model_name] = model_results

                # Save partial results after each model
                self.save_partial_results(results)

            except Exception as e:
                logging.error(f"Error analyzing with {model_name}: {e}")
                results["model_analysis"][model_name] = {"error": str(e)}

        return results

    def save_partial_results(self, results: Dict[str, Any], output_dir: str = "partial_results"):
        """Save partial results to avoid data loss."""
        Path(output_dir).mkdir(exist_ok=True)
        output_file = Path(output_dir) / f"partial_{results['original_id']}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        logging.info(f"Saved partial results for conversation {results['original_id']}")


def main():
    parser = argparse.ArgumentParser(description="Analyze conversations using multiple models")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to input JSON file containing conversations")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--models", nargs='+', required=True,
                        help="List of model names to use for analysis")

    try:
        args = parser.parse_args()
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load conversations
        with open(args.input_file, 'r', encoding='utf-8') as f:
            conversations = json.load(f)

        # Initialize analyzer
        analyzer = ConversationAnalyzer(args.models)

        # Process each conversation
        all_results = []
        for conv in conversations:
            print(conv)
            result = analyzer.analyze_conversation(conv)
            all_results.append(result)

            # Save complete results after each conversation
            output_file = output_dir / f"complete_results.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=4)

            logging.info(f"Processed conversation {conv['conversation_id']}")
    except Exception as e:
        logging.error(f"Error during processing: {e}")


if __name__ == "__main__":
    main()
