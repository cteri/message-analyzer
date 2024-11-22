import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import ollama

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

questions = """
[Question1]. Has any person given their age? (and what age was given)
[Question2]. Has any person asked the other for their age?
[Question3]. Has any person asked to meet up in person? Where?
[Question4]. Has any person given a gift to the other? Or bought something from a list like an amazon wish list?
[Question5]. Have any videos or photos been produced? Requested?
"""


class LlamaModel:
    # def __init__(self, model_name="llama3.2:1b"):
    def __init__(self, model_name="llama3.1"):
        self.model_name = model_name
        self.client = ollama
        self.chunk_size = 64000
        self.chunk_overlap = 100
        self.conversation_history = []

    def _generate_response(self, prompt: str) -> str:
        """Generate response using Ollama."""
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                stream=False,
                options={"temperature": 0.6, "top_p": 0.9, "num_predict": 5000},
            )
            return response["response"]
        except Exception as e:
            logging.error(f"Error generating response from Ollama: {e}")
            raise

    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            if end < len(text):
                # Try to find a natural break point
                split_point = text.rfind("\n", start, end)
                if split_point == -1:
                    split_point = text.rfind(". ", start, end)
                if split_point != -1:
                    end = split_point + 1
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap
        return chunks

    def _create_prompt(self, conversation_chunk: str) -> str:
        """Create analysis prompt with context."""
        template = f"""
        You are analyzing a conversation for specific patterns and behaviors. 
        Please review the following conversation carefully and answer each question precisely.

        Previous context: {self.conversation_history[-3:] if self.conversation_history else 'No previous context'}
        Current conversation segment: {conversation_chunk}

        For each question below, provide a clear YES/NO answer followed by specific evidence or "No evidence found":

        {questions}

        Provide your analysis in the following JSON format:
        {{
            "analysis": {{
                "questions": [
                    {{
                        "question_number": "1",
                        "question": "Has any person given their age? (and what age was given)",
                        "answer": "NO",
                        "evidence": "No evidence found in conversation"
                    }},
                    ...
                ]
            }}
        }}

        Instructions:
        - Answer must be either "YES" or "NO"
        - If answer is "YES": Provide specific evidence quotes
        - If answer is "NO": Set evidence to "No evidence found in conversation"
        - Ensure output is valid JSON format
        """
        return template

    def ask_questions(self, conversation_data: List[Dict]) -> List[Dict]:
        conversation_text = "\n".join(str(conv) for conv in conversation_data)
        chunks = self._split_text(conversation_text)
        all_results = []

        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for chunk in chunks:
                prompt = self._create_prompt(chunk)
                futures.append(executor.submit(self._generate_response, prompt))
                self.conversation_history.append(chunk)

            for future in futures:
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    logging.error(f"Error processing chunk: {e}")

        return all_results

    def load_data(self, file_path: str) -> List[Dict]:
        """Load data from file."""
        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".json":
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            elif ext in [".txt", ".csv"]:
                with open(file_path, "r", encoding="utf-8") as f:
                    return [{"conversation_text": f.read()}]
            else:
                logging.error(f"Unsupported file format: {ext}")
                return []
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            return []

    def clean_and_format_response(self, all_results, file_path: str) -> Dict:
        """
        Format analysis results into a standardized JSON structure with empty defaults.
        """
        try:
            # Initialize the base structure with empty values
            formatted_output = {
                "file_path": file_path,
                "conversation_ids": [],
                "analysis": {"questions": []},
            }

            # Define the standard questions structure
            standard_questions = [
                "Has any person given their age? (and what age was given)",
                "Has any person asked the other for their age?",
                "Has any person asked to meet up in person? Where?",
                "Has any person given a gift to the other? Or bought something from a list like an amazon wish list?",
                "Have any videos or photos been produced? Requested?",
            ]

            # Initialize questions with empty format
            for idx, question in enumerate(standard_questions, 1):
                question_entry = {
                    "question_number": str(idx),
                    "question": question,
                    "answer": "",
                    "evidence": "",
                    "instances": [],
                }
                formatted_output["analysis"]["questions"].append(question_entry)

            # Process each result from the model
            for result in all_results:
                try:
                    # Parse the JSON response
                    result_data = None
                    if isinstance(result, str):
                        # Extract JSON string between backticks if present
                        if "```json" in result:
                            json_str = (
                                result.split("```json")[1].split("```")[0].strip()
                            )
                            result_data = json.loads(json_str)
                        else:
                            result_data = json.loads(result)
                    else:
                        result_data = result

                    # Update formatted output with any evidence or instances
                    if (
                        result_data
                        and "analysis" in result_data
                        and "questions" in result_data["analysis"]
                    ):
                        for new_q in result_data["analysis"]["questions"]:
                            q_num = new_q.get("question_number")
                            if not q_num:
                                continue

                            idx = int(q_num) - 1
                            if idx >= len(formatted_output["analysis"]["questions"]):
                                continue

                            current_q = formatted_output["analysis"]["questions"][idx]

                            # Update answer and evidence if present
                            if new_q.get("answer"):
                                current_q["answer"] = new_q["answer"]
                            if new_q.get("evidence"):
                                current_q["evidence"] = new_q["evidence"]

                            # Add new instances if they exist and are not duplicates
                            new_instances = new_q.get("instances", [])
                            if new_instances:
                                existing_instances = {
                                    (inst.get("speaker", ""), inst.get("message", ""))
                                    for inst in current_q["instances"]
                                }

                                for inst in new_instances:
                                    instance_key = (
                                        inst.get("speaker", ""),
                                        inst.get("message", ""),
                                    )
                                    if instance_key not in existing_instances:
                                        current_q["instances"].append(inst)
                                        existing_instances.add(instance_key)

                except json.JSONDecodeError as je:
                    logging.warning(f"Failed to parse result JSON: {je}")
                    continue
                except Exception as e:
                    logging.warning(f"Error processing result: {e}")
                    continue

            # Post-processing: Set default values for empty fields
            for question in formatted_output["analysis"]["questions"]:
                if not question["answer"]:
                    question["answer"] = "NO"
                if not question["evidence"]:
                    question["evidence"] = "No evidence found in conversation"

            return formatted_output

        except Exception as e:
            logging.error(f"Error formatting response: {e}")
            # Return empty structure with error information
            return {
                "file_path": file_path,
                "conversation_ids": [],
                "error": str(e),
                "analysis": {
                    "questions": [
                        {
                            "question_number": str(i + 1),
                            "question": q,
                            "answer": "",
                            "evidence": "",
                            "instances": [],
                        }
                        for i, q in enumerate(standard_questions)
                    ]
                },
            }

    def analysis(self, file_paths: List[str]) -> List[Dict]:
        """
        Analyze multiple conversation files and return formatted results.
        """
        results = []
        for file_path in file_paths:
            try:
                data = self.load_data(file_path)
                if not data:
                    logging.error(f"No data loaded for {file_path}.")
                    continue

                analysis_results = self.ask_questions(data)
                formatted_result = self.clean_and_format_response(
                    analysis_results, file_path
                )

                results.append({"file_path": file_path, "result": formatted_result})

            except Exception as e:
                logging.error(f"Error analyzing file {file_path}: {str(e)}")
                results.append(
                    {
                        "file_path": file_path,
                        "error": str(e),
                        "result": {
                            "conversation_ids": ["unknown"],
                            "analysis": {"questions": []},
                        },
                    }
                )

        return results
