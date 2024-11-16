import os
import json
import logging
from typing import List, Dict, Any
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from huggingface_hub import login

token = os.getenv('HUGGING_FACE_TOKEN')
login(token=token)

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

questions = """
[Question1]. Has any person given their age? (and what age was given)
[Question2]. Has any person asked the other for their age?
[Question3]. Has any person asked to meet up in person? Where?
[Question4]. Has any person given a gift to the other? Or bought something from a list like an amazon wish list?
[Question5]. Have any videos or photos been produced? Requested?
"""


class LlamaModel:
    # def __init__(self, model_id="meta-llama/Llama-3.1-8B-Instruct"):
    def __init__(self, model_id="meta-llama/Llama-3.2-1B-Instruct"):
        self.model_id = model_id
        self.llm = self._setup_llm()
        # Increased chunk size and reduced overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=64000,
            chunk_overlap=100,
            length_function=len
        )
        self.memory = ConversationBufferMemory(
            input_key="conversation_chunk",
            memory_key="chat_history",
            return_messages=True
        )
        self._setup_chain()

    def _setup_llm(self) -> HuggingFacePipeline:
        """Setup LLaMA model with LangChain."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            model_kwargs = {
                "torch_dtype": torch.float16,
                "low_cpu_mem_usage": True,
            }

            # Check available GPU memory and adjust accordingly
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3  # Convert to GB
                if gpu_memory < 8:  # If less than 8GB GPU memory
                    model_kwargs["device_map"] = "auto"
                    model_kwargs["offload_folder"] = "offload"
                    os.makedirs("offload", exist_ok=True)
                else:
                    model_kwargs["device_map"] = "cuda:0"
            else:
                # CPU-only setup with memory efficient loading
                model_kwargs["device_map"] = None

            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **model_kwargs
            )

            # Configure pipeline with appropriate settings
            pipe_kwargs = {
                "max_new_tokens": 5000,
                "temperature": 0.6,
                "top_p": 0.9,
            }

            if torch.cuda.is_available():
                pipe_kwargs["device"] = "cuda"

            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                **pipe_kwargs
            )

            return HuggingFacePipeline(pipeline=pipe)
        except Exception as e:
            logging.error(f"Error setting up model: {e}")
            raise


    def _setup_chain(self):
        """Setup the QA chain with custom prompt."""
        template = """
            You are analyzing a conversation for specific patterns and behaviors. 
            Please review the following conversation carefully and answer each question precisely.
    
            Previous context: {chat_history}
            Current conversation segment: {conversation_chunk}
    
            For each question below, provide a clear YES/NO answer followed by specific evidence or "No evidence found":
    
            {questions}
    
            You are analyzing a conversation for specific patterns and behaviors. Review the conversation carefully and provide your analysis in the following JSON format:
            ```json
            {{
                "analysis": {{
                    "questions": [
                        {{
                            "question_number": "1",
                            "question": "Has any person given their age? (and what age was given)",
                            "answer": "NO",
                            "evidence": "No evidence found in conversation",
                            "instances": []
                        }},
                        {{
                            "question_number": "2",
                            "question": "Has any person asked the other for their age?",
                            "answer": "NO",
                            "evidence": "No evidence found in conversation",
                            "instances": []
                        }},
                        {{
                            "question_number": "3",
                            "question": "Has any person asked to meet up in person? Where?",
                            "answer": "NO",
                            "evidence": "No evidence found in conversation",
                            "instances": []
                        }},
                        {{
                            "question_number": "4",
                            "question": "Has any person given a gift to the other? Or bought something from a list like an amazon wish list?",
                            "answer": "NO",
                            "evidence": "No evidence found in conversation",
                            "instances": []
                        }},
                        {{
                            "question_number": "5",
                            "question": "Have any videos or photos been produced? Requested?",
                            "answer": "NO",
                            "evidence": "No evidence found in conversation",
                            "instances": []
                        }}
                    ]
                }}
            }}
            ```
            Instructions:
    
            Answer must be either "YES" or "NO"
            If answer is "YES": Provide specific evidence quotes in the "evidence" field
            If answer is "NO": Set evidence to "No evidence found in conversation"
    
            Ensure the output is valid JSON format and includes all required fields.
            """

        self.prompt = PromptTemplate(
            input_variables=["chat_history", "conversation_chunk", "questions"],
            template=template
        )

        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory,
            verbose=True
        )


    def ask_questions(self, conversation_data: List[Dict]) -> List[Dict]:
        conversation_text = "\n".join(
            str(conv.get('conversation_text', str(conv)))
            for conv in conversation_data
        )

        chunks = self.text_splitter.split_text(conversation_text)
        all_results = []

        # Process chunks in batches
        batch_size = 4
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            with torch.cuda.amp.autocast():  # Added automatic mixed precision
                for chunk in batch_chunks:
                    chain_input = {
                        "conversation_chunk": chunk,
                        "questions": questions
                    }
                    result = self.chain(chain_input)["text"]
                    all_results.append(result)

        return all_results


    def load_data(self, file_path: str) -> List[Dict]:
        """Compatible with original interface - loads and processes data."""
        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif ext in ['.txt', '.csv']:
                with open(file_path, 'r', encoding='utf-8') as f:
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

        Args:
            all_results: List of raw analysis results from model
            file_path: Path of the analyzed file

        Returns:
            Dict with standardized format containing analysis results
        """
        try:
            # Initialize the base structure with empty values
            formatted_output = {
                "file_path": file_path,
                "conversation_ids": [],
                "analysis": {
                    "questions": []
                }
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
                    "answer": "",
                    "evidence": "",
                    "instances": []
                }
                formatted_output["analysis"]["questions"].append(question_entry)

            # Process each result from the model
            for result in all_results:
                try:
                    # Parse the JSON response
                    result_data = None
                    if isinstance(result, str):
                        # Extract JSON string between backticks if present
                        if '```json' in result:
                            json_str = result.split('```json')[1].split('```')[0].strip()
                            result_data = json.loads(json_str)
                        else:
                            result_data = json.loads(result)
                    else:
                        result_data = result

                    # Update formatted output with any evidence or instances
                    if result_data and "analysis" in result_data and "questions" in result_data["analysis"]:
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
                                    instance_key = (inst.get("speaker", ""), inst.get("message", ""))
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
                            "instances": []
                        }
                        for i, q in enumerate(standard_questions)
                    ]
                }
            }


    def analysis(self, file_paths: List[str]) -> List[Dict]:
        """
        Analyze multiple conversation files and return formatted results.

        Args:
            file_paths: List of paths to conversation files

        Returns:
            List of dictionaries containing analysis results for each file
        """
        results = []
        for file_path in file_paths:
            try:
                # Load and validate data
                data = self.load_data(file_path)
                if not data:
                    logging.error(f"No data loaded for {file_path}.")
                    continue

                # Get analysis results
                analysis_results = self.ask_questions(data)

                # Clean and format the results
                formatted_result = self.clean_and_format_response(analysis_results, file_path)

                # Add to results list
                results.append({
                    "file_path": file_path,
                    "result": formatted_result
                })

            except Exception as e:
                logging.error(f"Error analyzing file {file_path}: {str(e)}")
                results.append({
                    "file_path": file_path,
                    "result": {
                        "conversation_ids": ["unknown"],
                        "error": str(e),
                        "analysis": {
                            "questions": []
                        }
                    }
                })

        return results
