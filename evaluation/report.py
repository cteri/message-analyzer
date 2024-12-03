'''
python3 evaluation/report.py --labeled-data "src/data_processing/cornell_movie_dialogs/labeled_csv/labeled_data_1-1000.csv" --conv-pattern "evaluation/conversations_part_*.csv" --output "evaluation/results.csv"
'''

import pandas as pd
import glob
from typing import Dict, List, Tuple
import os
import argparse
from collections import defaultdict

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


import pandas as pd
import glob
from typing import Dict, List, Tuple
import os
import argparse
from collections import defaultdict


# [前面的 load_labeled_data 和 load_conversation_file 函數保持不變]

def calculate_metrics_all_cases(true_pos: int, false_pos: int, false_neg: int, true_neg: int,
                                total: int = 1000) -> Dict:
    """Calculate metrics using all 1000 cases as denominator"""
    accuracy = (true_pos + true_neg) / total
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'true_positive': true_pos,
        'false_positive': false_pos,
        'true_negative': true_neg,
        'false_negative': false_neg,
        'total_cases': total
    }


def calculate_metrics_yes_only(true_pos: int, false_neg: int, total_yes: int) -> Dict:
    """Calculate metrics using only yes cases as denominator"""
    accuracy = true_pos / total_yes if total_yes > 0 else 0
    return {
        'accuracy': accuracy * 100,
        'true_positive': true_pos,
        'false_negative': false_neg,
        'total_yes_cases': total_yes
    }


def analyze_questions(merged_df: pd.DataFrame) -> Tuple[Dict, Dict, Dict]:
    """Analyze each question separately and return metrics along with raw numbers"""
    all_metrics = {}
    yes_only_metrics = {}
    raw_counts = {}

    for q_num in range(1, 6):
        q_label = f'Q{q_num}'
        labeled_col = f'{q_label}_labeled'
        conv_col = f'{q_label}_conv'

        # Get raw counts first
        yes_cases = merged_df[merged_df[labeled_col].str.lower().str.startswith('yes')]
        no_cases = merged_df[merged_df[labeled_col].str.lower() == 'no']

        true_pos = len(merged_df[(merged_df[labeled_col].str.lower().str.startswith('yes')) &
                                 (merged_df[conv_col].str.lower().str.startswith('yes'))])
        true_neg = len(merged_df[(merged_df[labeled_col].str.lower() == 'no') &
                                 (merged_df[conv_col].str.lower() == 'no')])
        false_pos = len(merged_df[(merged_df[labeled_col].str.lower() == 'no') &
                                  (merged_df[conv_col].str.lower().str.startswith('yes'))])
        false_neg = len(merged_df[(merged_df[labeled_col].str.lower().str.startswith('yes')) &
                                  (merged_df[conv_col].str.lower() == 'no')])

        total_yes = len(yes_cases)

        # Store raw counts
        raw_counts[q_label] = {
            'total_cases': len(merged_df),
            'total_yes_cases': total_yes,
            'total_no_cases': len(no_cases),
            'true_positive': true_pos,
            'true_negative': true_neg,
            'false_positive': false_pos,
            'false_negative': false_neg
        }

        # Calculate metrics with all cases
        all_metrics[q_label] = calculate_metrics_all_cases(
            true_pos, false_pos, false_neg, true_neg, len(merged_df)
        )

        # Calculate metrics with only yes cases
        yes_only_metrics[q_label] = calculate_metrics_yes_only(
            true_pos, false_neg, total_yes
        )

    return all_metrics, yes_only_metrics, raw_counts


def create_results_tables(all_metrics: Dict, yes_only_metrics: Dict, raw_counts: Dict) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create three separate tables for results"""
    # Raw counts table
    raw_data = []
    for q_num in range(1, 6):
        q_label = f'Q{q_num}'
        counts = raw_counts[q_label]
        raw_data.append({
            'Question': q_label,
            'Total_Cases': counts['total_cases'],
            'Total_Yes_Cases': counts['total_yes_cases'],
            'Total_No_Cases': counts['total_no_cases'],
            'True_Positive': counts['true_positive'],
            'True_Negative': counts['true_negative'],
            'False_Positive': counts['false_positive'],
            'False_Negative': counts['false_negative']
        })

    # All cases metrics table
    all_cases_results = []
    for q_num in range(1, 6):
        q_label = f'Q{q_num}'
        metrics = all_metrics[q_label]
        all_cases_results.append({
            'Question': q_label,
            'Accuracy_All': f"{metrics['accuracy']:.2f}%",
            'Precision_All': f"{metrics['precision']:.2f}%",
            'Recall_All': f"{metrics['recall']:.2f}%",
            'F1_All': f"{metrics['f1']:.2f}%"
        })

    # Yes only metrics table
    yes_only_results = []
    for q_num in range(1, 6):
        q_label = f'Q{q_num}'
        metrics = yes_only_metrics[q_label]
        yes_only_results.append({
            'Question': q_label,
            'Total_Yes_Cases': metrics['total_yes_cases'],
            'Correct_Yes_Cases': metrics['true_positive'],
            'Accuracy_Yes_Only': f"{metrics['accuracy']:.2f}%"
        })

    return (pd.DataFrame(raw_data),
            pd.DataFrame(all_cases_results),
            pd.DataFrame(yes_only_results))


def main():
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
        all_merged_data = []

        print("\nProcessing Files:")
        print("=" * 80)

        conversation_files = glob.glob(args.conv_pattern)
        if not conversation_files:
            print(f"Warning: No files found matching pattern: {args.conv_pattern}")
            return

        for file in conversation_files:
            print(f"\nProcessing file: {file}")
            try:
                conv_df = load_conversation_file(file)
                merged_df = pd.merge(labeled_df, conv_df, on='id', how='inner',
                                     suffixes=('_labeled', '_conv'))
                all_merged_data.append(merged_df)
            except Exception as e:
                print(f"Error processing file {file}: {str(e)}")
                continue

        if all_merged_data:
            combined_df = pd.concat(all_merged_data)
            all_metrics, yes_only_metrics, raw_counts = analyze_questions(combined_df)

            # Create results tables
            raw_df, all_cases_df, yes_only_df = create_results_tables(
                all_metrics, yes_only_metrics, raw_counts
            )

            # Display results
            print("\nRaw Counts:")
            print("=" * 100)
            print(raw_df.to_string(index=False))

            print("\nAll Cases Metrics:")
            print("=" * 100)
            print(all_cases_df.to_string(index=False))

            print("\nYes Only Metrics:")
            print("=" * 100)
            print(yes_only_df.to_string(index=False))

            # Save results
            raw_df.to_csv(args.output.replace('.csv', '_raw_counts.csv'), index=False)
            all_cases_df.to_csv(args.output.replace('.csv', '_all_cases.csv'), index=False)
            yes_only_df.to_csv(args.output.replace('.csv', '_yes_only.csv'), index=False)
            combined_df.to_csv(args.output.replace('.csv', '_merged.csv'), index=False)

            print(f"\nResults have been saved to separate CSV files")

        else:
            print("\nNo results were generated. Please check your input files and patterns.")

    except Exception as e:
        print(f"Error: {str(e)}")
        return


if __name__ == "__main__":
    main()