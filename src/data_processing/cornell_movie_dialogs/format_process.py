import csv
import json
from pathlib import Path

import pandas as pd


def load_all_conversations(current_dir):
    """Load conversations from multiple json files"""
    all_conversations = []

    # Load conversations from part 001 to part 010
    for i in range(1, 11):
        file_name = f"split_conversations/conversations_part_{i:03d}.json"
        json_path = current_dir / file_name

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                conversations = json.load(f)
                all_conversations.extend(conversations)
                print(f"Loaded {len(conversations)} conversations from {file_name}")
        except Exception as e:
            print(f"Error loading {file_name}: {str(e)}")

    print(f"Total conversations loaded: {len(all_conversations)}")
    return all_conversations


def process_data(labeled_df, conversations_list):
    # Create a dictionary to store conversations by ID
    conv_dict = {conv["conversation_id"]: conv for conv in conversations_list}

    # Create lists to store results
    results = []

    # Process each row in labeled data
    for _, row in labeled_df.iterrows():
        conv_id = row["ID"]

        if conv_id in conv_dict:
            conv = conv_dict[conv_id]

            # Create formatted output
            formatted_result = {
                "original_id": conv_id,
                "dialogue": [
                    {"speaker": turn["speaker"], "text": turn["text"]}
                    for turn in conv["turns"]
                ],
                "questions": {
                    "Q1": str(row["Q1: Age given"]),
                    "Q2": str(row["Q2: Age asked"]),
                    "Q3": str(row["Q3: Meet up request"]),
                    "Q4": str(row["Q4: Gift/Purchase"]),
                    "Q5": str(row["Q5: Videos/Photos"]),
                },
            }

            # Check if any question has a "Yes" answer
            has_yes = any(
                str(row[f"Q{i}: {q}"]).startswith("Yes")
                for i, q in enumerate(
                    [
                        "Age given",
                        "Age asked",
                        "Meet up request",
                        "Gift/Purchase",
                        "Videos/Photos",
                    ],
                    1,
                )
            )

            if has_yes:
                results.append(formatted_result)

    return results


def save_results(results, output_file="formatted_conversations.json"):
    # Write results to JSON with proper formatting
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    try:
        # Set paths
        current_dir = Path.cwd()
        labeled_data_path = current_dir / "labeled_data_1-1000.csv"

        print(f"Looking for labeled data in: {labeled_data_path}")
        print(f"Current working directory: {current_dir}")

        # Load labeled data
        labeled_data = pd.read_csv(labeled_data_path)
        print(f"Loaded labeled data with {len(labeled_data)} rows")

        # Load all conversations from multiple files
        conversations = load_all_conversations(current_dir)

        # Process the data
        results = process_data(labeled_data, conversations)

        # Save results
        output_file = current_dir / "formatted_conversations.json"
        save_results(results, output_file)

        print(
            f"\nAnalysis complete. Found {len(results)} conversations with 'Yes' answers."
        )
        print(f"Results have been saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback

        print(traceback.format_exc())


if __name__ == "__main__":
    main()
