'''
python3 evaluation/report.py --labeled-data "src/data_processing/cornell_movie_dialogs/labeled_csv/labeled_data_1-1000.csv" --conv-pattern "evaluation/conversations_part_*.csv" --output "evaluation/results.csv"
'''

import pandas as pd
import glob
from typing import Dict, Tuple
import os
import argparse


def load_labeled_data(filepath: str) -> pd.DataFrame:
    """Load and prepare the labeled data file"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Labeled data file not found: {filepath}")

    labeled_df = pd.read_csv(filepath)
    column_mapping = {
        'ID': 'id',
        'Q1: Age given': 'Q1',
        'Q2: Age asked': 'Q2',
        'Q3: Meet up request': 'Q3',
        'Q4: Gift/Purchase': 'Q4',
        'Q5: Videos/Photos': 'Q5'
    }
    labeled_df = labeled_df.rename(columns=column_mapping)
    return labeled_df


def load_conversation_file(filepath: str) -> pd.DataFrame:
    """Load and prepare a conversation data file"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Conversation file not found: {filepath}")
    return pd.read_csv(filepath)


def compare_answers(labeled_val: str, conv_val: str) -> bool:
    """Compare a single pair of answers"""
    labeled_val = str(labeled_val).strip().lower()
    conv_val = str(conv_val).strip().lower()

    # Handle special cases
    if labeled_val == "no" and conv_val == "no":
        return True
    elif labeled_val.startswith("yes") and conv_val.startswith("yes"):
        return True
    else:
        return labeled_val == conv_val


def check_all_answers(row: pd.Series) -> bool:
    """Check if all Q1-Q5 answers are correct"""
    return all(compare_answers(row[f'Q{i}_labeled'], row[f'Q{i}_conv'])
               for i in range(1, 6))


def analyze_file(conv_filepath: str, labeled_df: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Analyze a single conversation file and return statistics"""
    conv_df = load_conversation_file(conv_filepath)

    # Merge dataframes on id
    merged_df = pd.merge(
        labeled_df,
        conv_df,
        on='id',
        how='inner',
        suffixes=('_labeled', '_conv')
    )

    total_rows = len(merged_df)
    if total_rows == 0:
        print(f"Warning: No matching rows found for {conv_filepath}")
        return {}, pd.DataFrame()

    # Add a column indicating if all answers are correct
    merged_df['all_correct'] = merged_df.apply(check_all_answers, axis=1)

    # Calculate statistics
    total_correct = merged_df['all_correct'].sum()

    stats = {
        'total_accuracy': (total_correct / total_rows * 100) if total_rows > 0 else 0,
        'total_correct': total_correct,
        'total_rows': total_rows
    }

    return stats, merged_df


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze conversation data against labeled data.')
    parser.add_argument('--labeled-data', type=str, required=True,
                        help='Path to the labeled data CSV file')
    parser.add_argument('--conv-pattern', type=str, required=True,
                        help='Pattern to match conversation files (e.g., "conversations_part_*.csv")')
    parser.add_argument('--output', type=str, default='detailed_results.csv',
                        help='Path for output CSV file (default: detailed_results.csv)')

    args = parser.parse_args()

    try:
        labeled_df = load_labeled_data(args.labeled_data)
        all_results = []
        all_merged_data = []

        print("\nProcessing Files:")
        print("=" * 80)

        # Use the provided pattern to find conversation files
        conversation_files = glob.glob(args.conv_pattern)
        if not conversation_files:
            print(f"Warning: No files found matching pattern: {args.conv_pattern}")
            return

        for file in conversation_files:
            print(f"\nProcessing file: {file}")
            try:
                stats, merged_df = analyze_file(file, labeled_df)
                if stats:
                    all_results.append(stats)
                    all_merged_data.append(merged_df)

                    print(f"\nResults for {file}")
                    print("-" * 50)
                    print(f"Accuracy: {stats['total_accuracy']:.2f}% "
                          f"({stats['total_correct']}/{stats['total_rows']} correct)")
            except Exception as e:
                print(f"Error processing file {file}: {str(e)}")
                continue

        if all_results:
            # Combine all evaluation
            total_correct = sum(r['total_correct'] for r in all_results)
            total_rows = sum(r['total_rows'] for r in all_results)

            print("\nOverall Results:")
            print("=" * 80)
            print(f"Final Accuracy: {(total_correct / total_rows * 100):.2f}% "
                  f"({total_correct}/{total_rows} correct)")

            # Save detailed evaluation
            combined_df = pd.concat(all_merged_data)
            combined_df.to_csv(args.output, index=False)
            print(f"\nDetailed evaluation have been saved to '{args.output}'")
        else:
            print("\nNo results were generated. Please check your input files and patterns.")

    except Exception as e:
        print(f"Error: {str(e)}")
        return


if __name__ == "__main__":
    main()
