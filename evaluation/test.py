import pandas as pd
import glob
from typing import Dict, Tuple
import os


def load_labeled_data(filepath: str) -> pd.DataFrame:
    """Load and prepare the labeled data file"""
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
    labeled_data_path = '../src/data_processing/cornell_movie_dialogs/labeled_csv/labeled_data_1-1000.csv'

    if not os.path.exists(labeled_data_path):
        print(f"Error: {labeled_data_path} not found!")
        return

    labeled_df = load_labeled_data(labeled_data_path)
    all_results = []
    all_merged_data = []

    print("\nProcessing Files:")
    print("=" * 80)

    for file in glob.glob('conversations_part_*.csv'):
        print(f"\nProcessing file: {file}")
        stats, merged_df = analyze_file(file, labeled_df)
        if stats:
            all_results.append(stats)
            all_merged_data.append(merged_df)

            print(f"\nResults for {file}")
            print("-" * 50)
            print(f"Accuracy: {stats['total_accuracy']:.2f}% "
                  f"({stats['total_correct']}/{stats['total_rows']} correct)")

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
        combined_df.to_csv('detailed_results.csv', index=False)
        print("\nDetailed evaluation have been saved to 'detailed_results.csv'")


if __name__ == "__main__":
    main()