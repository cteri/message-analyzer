# csv_analysis_client.py

import argparse
import pandas as pd
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from src.ml.model import LlamaModel  # Import the existing model class

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class CSVAnalysisClient:
    def __init__(self, model: LlamaModel):
        self.model = model

    def prepare_conversation_data(self, row: pd.Series) -> Dict[str, str]:
        """Convert DataFrame row to model-compatible format

        Args:
            row: DataFrame row containing conversation data

        Returns:
            Dictionary containing formatted conversation text
        """
        # Construct conversation text format
        conversation_text = f"""
        Conversation ID: {row.name}
        Age Information:
        - Age Given: {row['Q1: Age given']}
        - Age Asked: {row['Q2: Age asked']}

        Interaction Details:
        - Meet Up Request: {row['Q3: Meet up request']}
        - Gift/Purchase: {row['Q4: Gift/Purchase']}
        - Videos/Photos: {row['Q5: Videos/Photos']}
        """

        return {"conversation_text": conversation_text, "id": row.name}

    def process_batch(self, df: pd.DataFrame, batch_size: int) -> List[Dict[str, Any]]:
        """Process conversations in batches

        Args:
            df: DataFrame containing all conversation data
            batch_size: Number of conversations to process per batch

        Returns:
            List of processing results
        """
        all_results = []
        total_batches = len(df) // batch_size + (1 if len(df) % batch_size > 0 else 0)

        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            logging.info(f"Processing batch {i // batch_size + 1}/{total_batches}")

            # Convert batch data to model-compatible format
            conversation_data = [
                self.prepare_conversation_data(row)
                for _, row in batch.iterrows()
            ]

            # Use existing model for analysis
            try:
                batch_results = self.model.ask_questions(conversation_data)
                all_results.extend(batch_results)
            except Exception as e:
                logging.error(f"Error processing batch: {e}")
                continue

        return all_results


def main():
    parser = argparse.ArgumentParser(description="Analyze conversation CSV data using LlamaModel")
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to input CSV file",
        required=True
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to save analysis results",
        required=True
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Number of conversations to process per batch"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model identifier to use"
    )

    args = parser.parse_args()

    try:
        # Load CSV data
        df = pd.read_csv(args.input_file, index_col="ID")
        logging.info(f"Loaded {len(df)} conversations from {args.input_file}")

        # Initialize model and client
        model = LlamaModel(model_id=args.model_id)
        client = CSVAnalysisClient(model)

        # Perform analysis
        results = client.process_batch(df, args.batch_size)

        # Add post-processing step to compare model analysis with original labels
        processed_results = {
            "analysis_summary": {
                "total_conversations": len(df),
                "processed_conversations": len(results),
                "timestamp": pd.Timestamp.now().isoformat()
            },
            "detailed_results": results,
            "validation_metrics": {
                "agreement_rate": {},
                "disagreement_details": []
            }
        }

        # Save results
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_results, f, indent=4, ensure_ascii=False)

        logging.info(f"Analysis complete. Results saved to {output_path}")

    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()