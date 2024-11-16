import argparse
import json
import warnings
from pathlib import Path
import logging
from src.ml.model import LlamaModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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

    print("\nRaw Response:")
    print(json.dumps(results, ensure_ascii=False, indent=2))

    # Print the complete results for debugging
    print("\n=== Complete Results ===")
    print(json.dumps(results, ensure_ascii=False, indent=2))

    # Save results
    for result in results:
        file_path = result["file_path"]
        print(f"\n=== Processing file: {file_path} ===")

        # Print all keys in the result
        print("\nAvailable keys in result:")
        print(list(result.keys()))

        analysis_results = result.get("result", {})
        print(analysis_results)
        raw_response = result.get("raw_response", {})

        print("\nAnalysis Results:")
        print(json.dumps(analysis_results, ensure_ascii=False, indent=2))

        # Prepare output file paths
        input_file_name = Path(file_path).stem
        output_file = output_directory / f"{input_file_name}_analysis.json"
        raw_output_file = output_directory / f"{input_file_name}_raw_response.json"

        try:
            # Save analysis results
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(analysis_results, f, ensure_ascii=False, indent=4)
            logging.info(f"Analysis for {file_path} saved to {output_file}")

            # Save raw response
            save_raw_response(raw_response, raw_output_file)

        except Exception as e:
            logging.error(f"Error saving results for {file_path}: {e}")
            print(f"\nError details: {str(e)}")


if __name__ == "__main__":
    main()
