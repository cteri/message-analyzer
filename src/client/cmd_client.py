import argparse
import json
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


def save_raw_response(response, output_path):
    """Save the raw model response to a file."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(response, f, ensure_ascii=False, indent=4)
            logging.info(f"Raw response saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving raw response: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze conversations using LlamaModel"
    )
    parser.add_argument(
        "--input_files",
        type=str,
        nargs="+",
        help="Paths to the input conversation files (supports .json, .txt, .csv)",
        required=True,
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        help="The path to the directory to save the analysis outputs",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="The name of the Ollama model to use",
        default="llama3.2:1b",
    )

    args = parser.parse_args()

    # Prepare the output directory
    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Initialize the model
    try:
        model = LlamaModel(model_name=args.model_name)
        logging.info(f"Successfully initialized model: {args.model_name}")
    except Exception as e:
        logging.error(f"Failed to initialize model: {e}")
        return

    # Perform analysis on the input files
    results = model.analysis(args.input_files)

    # Process and save results
    for result in results:
        file_path = result["file_path"]
        logging.info(f"\nProcessing file: {file_path}")

        analysis_results = result.get("result", {})

        # Prepare output file paths
        input_file_name = Path(file_path).stem
        output_file = output_directory / f"{input_file_name}_analysis.json"

        try:
            # Save analysis results
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(analysis_results, f, ensure_ascii=False, indent=4)
            logging.info(f"Analysis for {file_path} saved to {output_file}")

            # Log analysis summary
            if "analysis" in analysis_results:
                questions = analysis_results["analysis"].get("questions", [])
                for q in questions:
                    if q.get("answer") == "YES":
                        logging.info(
                            f"Found evidence for question {q['question_number']}: {q['evidence']}"
                        )

        except Exception as e:
            logging.error(f"Error saving results for {file_path}: {e}")
            logging.error(f"Error details: {str(e)}")


if __name__ == "__main__":
    main()
