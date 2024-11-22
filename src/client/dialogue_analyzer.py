import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from langchain.text_splitter import RecursiveCharacterTextSplitter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DialogueAnalyzer:
    def __init__(self, model):
        self.model = model
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=64000, chunk_overlap=100, length_function=len
        )

    def format_dialogue(self, conversation: Dict[str, Any]) -> str:
        """Format dialogue into a readable conversation format for the LLM

        Args:
            conversation: Dictionary containing dialogue and metadata

        Returns:
            Formatted conversation string
        """
        dialogue_text = f"\nConversation ID: {conversation['original_id']}\n"
        dialogue_text += "Dialogue:\n"

        for utterance in conversation["dialogue"]:
            dialogue_text += f"{utterance['speaker']}: {utterance['text']}\n"

        return dialogue_text

    def prepare_batch_prompt(self, conversations: List[Dict[str, Any]]) -> str:
        """Prepare a batch of conversations for analysis

        Args:
            conversations: List of conversation dictionaries

        Returns:
            Formatted prompt for the model
        """
        formatted_conversations = []
        for conv in conversations:
            formatted_conversations.append(self.format_dialogue(conv))

        # Join all formatted conversations
        conversation_text = "\n".join(formatted_conversations)

        return conversation_text

    def process_batch(self, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a batch of conversations and store raw responses

        Args:
            conversations: List of conversation dictionaries

        Returns:
            Raw model response and metadata
        """
        try:
            # Prepare conversation text
            conversation_text = self.prepare_batch_prompt(conversations)

            # Create data structure for model
            conversation_data = [{"conversation_text": conversation_text}]

            # Get model's analysis
            model_responses = self.model.ask_questions(conversation_data)

            # Format the response
            formatted_response = self.model.clean_and_format_response(
                model_responses, f"batch_{conversations[0]['original_id']}"
            )

            return {
                "conversations": [conv["original_id"] for conv in conversations],
                "raw_response": formatted_response,
                "original_labels": {
                    conv["original_id"]: conv.get("questions", {})
                    for conv in conversations
                },
                "timestamp": logging.Formatter().converter(),
                "status": "success",
            }

        except Exception as e:
            logging.error(f"Error in model analysis: {e}")
            return {
                "error": str(e),
                "conversations": [conv["original_id"] for conv in conversations],
                "timestamp": logging.Formatter().converter(),
                "status": "error",
            }


def analyze_raw_results(raw_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Analyze raw model responses to extract structured information

    Args:
        raw_results: List of raw batch processing results

    Returns:
        List of analyzed results
    """
    analyzed_results = []

    for batch_result in raw_results:
        if batch_result["status"] == "error":
            analyzed_results.append(batch_result)
            continue

        try:
            analyzed_batch = {
                "conversations": batch_result["conversations"],
                "original_labels": batch_result["original_labels"],
                "model_analysis": batch_result["raw_response"]["analysis"],
                "raw_response_id": batch_result.get("raw_response", {}).get(
                    "file_path"
                ),
                "status": "success",
            }
        except Exception as e:
            logging.error(f"Error analyzing batch result: {e}")
            analyzed_batch = {
                "error": str(e),
                "conversations": batch_result["conversations"],
                "status": "error",
            }

        analyzed_results.append(analyzed_batch)

    return analyzed_results


def compare_results(analyzed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare analyzed results with original labels

    Args:
        analyzed_results: List of analyzed batch results

    Returns:
        Comparison metrics and statistics
    """
    metrics = {
        "total_conversations": 0,
        "successful_analyses": 0,
        "failed_analyses": 0,
        "question_accuracy": {
            "Q1": {"correct": 0, "total": 0},
            "Q2": {"correct": 0, "total": 0},
            "Q3": {"correct": 0, "total": 0},
            "Q4": {"correct": 0, "total": 0},
            "Q5": {"correct": 0, "total": 0},
        },
        "discrepancies": [],
    }

    for batch_result in analyzed_results:
        if batch_result["status"] == "error":
            metrics["failed_analyses"] += len(batch_result["conversations"])
            continue

        metrics["successful_analyses"] += len(batch_result["conversations"])

        for conv_id, original_labels in batch_result["original_labels"].items():
            metrics["total_conversations"] += 1

            # Compare each question's result
            model_answers = batch_result["model_analysis"]["questions"]
            for i, (q_key, original_value) in enumerate(original_labels.items(), 1):
                metrics["question_accuracy"][f"Q{i}"]["total"] += 1
                model_answer = model_answers[i - 1]["answer"].lower()

                # Check if model's answer matches original label
                if bool(original_value) == bool("no" not in model_answer):
                    metrics["question_accuracy"][f"Q{i}"]["correct"] += 1
                else:
                    metrics["discrepancies"].append(
                        {
                            "conversation_id": conv_id,
                            "question": q_key,
                            "original_label": original_value,
                            "model_answer": model_answer,
                        }
                    )

    # Calculate accuracy percentages
    for q_stats in metrics["question_accuracy"].values():
        q_stats["accuracy"] = (
            (q_stats["correct"] / q_stats["total"]) if q_stats["total"] > 0 else 0
        )

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Analyze dialogue conversations using LLM"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to the formatted conversations JSON file",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save raw responses and analysis results",
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Number of conversations to process in each batch",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model identifier to use",
    )

    args = parser.parse_args()

    try:
        # Load conversations
        with open(args.input_file, "r", encoding="utf-8") as f:
            conversations = json.load(f)

        logging.info(
            f"Loaded {len(conversations)} conversations from {args.input_file}"
        )

        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model and analyzer
        from src.ml.model import LlamaModel

        model = LlamaModel(model_id=args.model_id)
        analyzer = DialogueAnalyzer(model)

        # Process in batches and store raw results
        raw_results = []
        for i in range(0, len(conversations), args.batch_size):
            batch = conversations[i : i + args.batch_size]
            logging.info(f"Processing batch {i // args.batch_size + 1}")

            batch_results = analyzer.process_batch(batch)
            raw_results.append(batch_results)

        # Save raw results
        raw_results_path = output_dir / "raw_results.json"
        with open(raw_results_path, "w", encoding="utf-8") as f:
            json.dump(raw_results, f, indent=4, ensure_ascii=False)

        logging.info(f"Raw results saved to {raw_results_path}")

        # Analyze raw results
        analyzed_results = analyze_raw_results(raw_results)

        # Compare results and generate metrics
        comparison_metrics = compare_results(analyzed_results)

        # Save analysis results
        analysis_output = {
            "analysis_summary": comparison_metrics,
            "detailed_results": analyzed_results,
        }

        analysis_path = output_dir / "analysis_results.json"
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(analysis_output, f, indent=4, ensure_ascii=False)

        logging.info(f"Analysis complete. Results saved to {analysis_path}")

    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
