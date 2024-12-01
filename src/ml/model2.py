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
    
    def process_folder(self, folder_path: str, max_files: int = 5) -> List[Dict]:
        results = []
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')][:max_files]
        
        for csv_file in csv_files:
            try:
                file_path = os.path.join(folder_path, csv_file)
                data = self.load_data(file_path)
                
                # Process single file
                response = self._generate_response(self._create_prompt(data[0]))
                
                # Parse response
                parsed_result = {
                    "file_path": csv_file,
                    "result": json.loads(response)
                }
                results.append(parsed_result)
                
            except Exception as e:
                logging.error(f"Error processing {csv_file}: {e}")
                continue
            logging.info(f"Processing files: {csv_files}")
            logging.info(f"Generated result: {parsed_result}")
                
        return results

    def _generate_response(self, prompt: str) -> str:
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                stream=False,
                options={"temperature": 0.6, "top_p": 0.9, "num_predict": 5000},
            )
            
            text = response["response"]
            
            # Extract JSON, handling all possible formats
            if "```" in text:
                parts = text.split("```")
                for part in parts:
                    if "{" in part and "}" in part:
                        start = part.find("{")
                        end = part.rfind("}") + 1
                        json_content = part[start:end]
                        break
            else:
                start = text.find("{")
                end = text.rfind("}") + 1
                json_content = text[start:end]

            # Pre-process to handle common issues
            json_content = (
                json_content
                .replace("\n", " ")
                .replace(r"\'", "'")
                .replace(r'\"', '"')
                .replace("\\\\", "\\")
            )
            
            # Parse and clean
            data = json.loads(json_content)
            for q in data["analysis"]["questions"]:
                if q.get("evidence"):
                    q["evidence"] = (
                        q["evidence"]
                        .strip('"')
                        .strip("'")
                        .replace(" and ", ". ")
                    )
            
            return json.dumps(data, ensure_ascii=False)
                
        except Exception as e:
            logging.error(f"Error with response: {text if 'text' in locals() else 'No response'}")
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
        """Create analysis prompt without context."""
        template = f"""
        You are analyzing a conversation for specific patterns and behaviors. 
        Please review the following conversation carefully and answer each question precisely.

        Current conversation segment: {conversation_chunk}

        For each question below, provide a clear YES/NO answer followed by specific evidence or "No evidence found".
        Important: When providing evidence, escape any quotes and special characters properly.

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
        - Ensure all quotes and special characters in evidence are properly escaped
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
                "analysis": {"questions": []}
            }

            # Define the standard questions structure
            standard_questions = [
                "Has any person given their age? (and what age was given)",
                "Has any person asked the other for their age?",
                "Has any person asked to meet up in person? Where?",
                "Has any person given a gift to the other? Or bought something from a list like an amazon wish list?",
                "Have any videos or photos been produced? Requested?"
            ]

            # Initialize questions with empty format
            for idx, question in enumerate(standard_questions, 1):
                question_entry = {
                    "question_number": str(idx),
                    "question": question,
                    "answer": "NO",
                    "evidence": "No evidence found in conversation",
                    "instances": []
                }
                formatted_output["analysis"]["questions"].append(question_entry)

            # Process each result from the model
            for result in all_results:
                try:
                    # If result is a string, try to parse it as JSON
                    if isinstance(result, str):
                        try:
                            result_data = json.loads(result)
                        except json.JSONDecodeError:
                            logging.warning(f"Failed to parse result as JSON: {result}")
                            continue
                    else:
                        result_data = result

                    # Extract questions from result
                    if (result_data and 
                        "analysis" in result_data and 
                        "questions" in result_data["analysis"]):
                        
                        for new_q in result_data["analysis"]["questions"]:
                            try:
                                q_num = int(new_q.get("question_number", "0")) - 1
                                if 0 <= q_num < len(formatted_output["analysis"]["questions"]):
                                    current_q = formatted_output["analysis"]["questions"][q_num]
                                    
                                    # Update answer if present
                                    if new_q.get("answer"):
                                        current_q["answer"] = new_q["answer"]
                                    
                                    # Update evidence if present and answer is YES
                                    if new_q.get("evidence") and new_q.get("answer") == "YES":
                                        current_q["evidence"] = new_q["evidence"]
                                    
                                    # Update instances if present
                                    if new_q.get("instances"):
                                        current_q["instances"].extend(new_q["instances"])
                            
                            except (ValueError, IndexError) as e:
                                logging.warning(f"Error processing question: {e}")
                                continue

                except Exception as e:
                    logging.warning(f"Error processing result: {e}")
                    continue

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
                            "answer": "NO",
                            "evidence": "Error processing results",
                            "instances": []
                        } for i, q in enumerate(standard_questions)
                    ]
                }
            }

    def analysis(self, file_paths: List[str]) -> List[Dict]:
        results = []
        for file_path in file_paths:
            try:
                data = self.load_data(file_path)
                response = self._generate_response(self._create_prompt(data[0]))
                results.append({
                    "file_path": os.path.basename(file_path),
                    "result": json.loads(response)
                })
            except Exception as e:
                logging.error(f"Error in file {file_path}: {e}")
                continue
        return results
