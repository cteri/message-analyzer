import argparse
import json
import warnings
from pathlib import Path
import logging

from src.ml.model import LlamaModel

# Ignore all warnings (if necessary)
warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description="Analyze conversations using LlamaModel")
    parser.add_argument(
        "--input_files",
        type=str,
        nargs='+',
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
        "--model_id",
        type=str,
        help="The identifier of the model to use",
        default="meta-llama/Llama-3.2-1B-Instruct",
    )
    args = parser.parse_args()

    # Prepare the output directory
    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Initialize the model
    model = LlamaModel(model_id=args.model_id)

    # Perform analysis on the input files
    results = model.analysis(args.input_files)

    # Save results
    for result in results:
        file_path = result["file_path"]
        analysis_results = result["result"]

        # Prepare output file path
        input_file_name = Path(file_path).stem
        output_file = output_directory / f"{input_file_name}_analysis.json"

        # Save analysis results to JSON file
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(analysis_results, f, ensure_ascii=False, indent=4)
            print(f"Analysis for {file_path} saved to {output_file}")
        except Exception as e:
            logging.error(f"Error saving results for {file_path}: {e}")


if __name__ == "__main__":
    main()
