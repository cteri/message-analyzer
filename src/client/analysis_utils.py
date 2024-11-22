# analysis_utils.py

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


class AnalysisValidator:
    """Utility class for validating and comparing analysis results"""

    @staticmethod
    def compare_with_original(
        original_df: pd.DataFrame, analysis_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare analysis results with original labels

        Args:
            original_df: Original data DataFrame
            analysis_results: List of model analysis results

        Returns:
            Dictionary containing comparison metrics
        """
        metrics = {
            "total_comparisons": 0,
            "matches": 0,
            "mismatches": [],
            "questions": {
                "age_given": {"correct": 0, "total": 0},
                "age_asked": {"correct": 0, "total": 0},
                "meetup": {"correct": 0, "total": 0},
                "gift": {"correct": 0, "total": 0},
                "media": {"correct": 0, "total": 0},
            },
        }

        for result in analysis_results:
            conv_id = result.get("conversation_ids", ["unknown"])[0]
            if conv_id in original_df.index:
                metrics["total_comparisons"] += 1
                original_row = original_df.loc[conv_id]

                # Compare results for each question
                analysis = result.get("analysis", {}).get("questions", [])
                for q in analysis:
                    question_num = int(q.get("question_number", 0))
                    if 1 <= question_num <= 5:
                        field_name = {
                            1: "age_given",
                            2: "age_asked",
                            3: "meetup",
                            4: "gift",
                            5: "media",
                        }[question_num]

                        # Update statistics
                        metrics["questions"][field_name]["total"] += 1
                        original_value = original_row[
                            f"Q{question_num}: {field_name.replace('_', ' ').title()}"
                        ]

                        if str(original_value).lower() in q.get("answer", "").lower():
                            metrics["questions"][field_name]["correct"] += 1
                        else:
                            metrics["mismatches"].append(
                                {
                                    "conversation_id": conv_id,
                                    "field": field_name,
                                    "original": original_value,
                                    "analysis": q.get("answer"),
                                }
                            )

        # Calculate overall accuracy
        total_correct = sum(q["correct"] for q in metrics["questions"].values())
        total_questions = sum(q["total"] for q in metrics["questions"].values())
        metrics["overall_accuracy"] = (
            total_correct / total_questions if total_questions > 0 else 0
        )

        return metrics


def load_and_validate_csv(file_path: str) -> pd.DataFrame:
    """Load and validate CSV file format

    Args:
        file_path: Path to CSV file

    Returns:
        Validated DataFrame

    Raises:
        ValueError: If required columns are missing
    """
    try:
        df = pd.read_csv(file_path)
        required_columns = [
            "ID",
            "Q1: Age given",
            "Q2: Age asked",
            "Q3: Meet up request",
            "Q4: Gift/Purchase",
            "Q5: Videos/Photos",
        ]

        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        return df.set_index("ID")

    except Exception as e:
        logging.error(f"Error loading CSV file: {e}")
        raise


def save_analysis_results(
    results: Dict[str, Any], output_path: str, include_validation: bool = True
) -> None:
    """Save analysis results

    Args:
        results: Analysis results dictionary
        output_path: Output file path
        include_validation: Whether to include validation metrics
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove validation metrics if not needed
        if not include_validation:
            results.pop("validation_metrics", None)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        logging.info(f"Results saved to {output_path}")

    except Exception as e:
        logging.error(f"Error saving results: {e}")
        raise
