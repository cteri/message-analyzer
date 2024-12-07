import argparse
import csv
import logging
import warnings
from pathlib import Path

from src.ml.model import LlamaModel

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Ignore all warnings
warnings.filterwarnings("ignore")


def format_answer(question):
    """Format the answer and evidence into a single string."""
    answer = question.get("answer", "")
    evidence = question.get("evidence", "")

    if answer == "YES" and evidence and evidence != "No evidence found in conversation":
        return f"YES - {evidence}"
    return answer


def main():
    parser = argparse.ArgumentParser(
        description="Analyze conversations using LlamaModel"
    )
    parser.add_argument(
        "--input_directory",
        type=str,
        help="Path to the directory containing conversation files",
        required=True,
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to the output CSV file",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="The name of the Ollama model to use",
        default="llama3.2:1b",
    )

    args = parser.parse_args()

    # Initialize the model and get results
    model = LlamaModel(model_name=args.model_name)
    input_files = list(Path(args.input_directory).glob("*"))
    results = model.analysis(input_files)

    # Create output directory if needed
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write results to CSV
    fieldnames = ["filename"] + [f"Q{i}" for i in range(1, 6)]

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            file_path = result["file_path"]
            analysis_results = result.get("result", {}).get("analysis", {})
            row = {"filename": Path(file_path).stem}

            questions = analysis_results.get("questions", [])
            for q in questions:
                question_number = q.get("question_number", "")
                row[f"Q{question_number}"] = format_answer(q)

            writer.writerow(row)

    logging.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
