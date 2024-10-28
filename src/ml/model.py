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
    def __init__(self, model_id="meta-llama/Llama-3.1-8B-Instruct"):
        self.model_id = model_id
        self.llm = self._setup_llm()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=64000,
            chunk_overlap=200,
            length_function=len
        )
        # Modified memory configuration to specify input key
        self.memory = ConversationBufferMemory(
            input_key="conversation_chunk",  # Specify which input we want to store
            memory_key="chat_history",
            return_messages=True
        )
        self._setup_chain()

    def _setup_llm(self) -> HuggingFacePipeline:
        """Setup LLaMA model with LangChain."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map="auto"
            )

            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=5000,
                temperature=0.6,
                top_p=0.9,
                device_map="auto"
            )

            return HuggingFacePipeline(pipeline=pipe)
        except Exception as e:
            logging.error(f"Error setting up model: {e}")
            raise

    def _setup_chain(self):
        """Setup the QA chain with custom prompt."""
        template = """
        Please carefully analyze the following conversation chunks and answer specific questions.
        Use the chat history and current conversation to provide accurate answers.

        Previous conversation chunks: {chat_history}
        Current conversation chunk: {conversation_chunk}

        Questions to answer:
        {questions}

        Please provide specific evidence from the conversations for each answer.
        If there's no relevant information, explicitly state so.

        Your analysis:
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

    def ask_questions(self, conversation_data: List[Dict]) -> List[Dict]:
        """Compatible with original interface - processes conversations and returns analysis."""
        try:
            # Convert conversation data to text
            conversation_text = "\n".join(
                str(conv.get('conversation_text', str(conv)))
                for conv in conversation_data
            )

            # Split text into chunks
            chunks = self.text_splitter.split_text(conversation_text)

            all_results = []

            # Process each chunk while maintaining conversation history
            for chunk in chunks:
                # Create a properly formatted input dictionary
                chain_input = {
                    "conversation_chunk": chunk,
                    "questions": questions
                }

                result = self.chain(chain_input)["text"]  # Use chain() instead of run()
                all_results.append(result)

            # Format the results
            formatted_results = self.clean_and_format_response(
                all_results=all_results,
                questions=questions
            )

            return [{
                "conversation_ids": [conv.get('id', 'unknown') for conv in conversation_data],
                "analysis": formatted_results
            }]

        except Exception as e:
            logging.error(f"Error in ask_questions: {e}")
            return [{"error": str(e)}]

    def clean_and_format_response(self, all_results: List[str], questions: str) -> Dict:
        """Format all chunk results into a single coherent response."""
        try:
            # Extract question pairs
            question_pairs = []
            for q in questions.split('\n'):
                if '[Question' in q:
                    marker = q[q.find('['):q.find(']') + 1]
                    question_text = q.replace(marker + '.', '').strip()
                    question_pairs.append((marker, question_text))

            # Combine all chunks' results
            formatted_output = {
                "questions": []
            }

            for marker, question_text in question_pairs:
                question_number = marker.replace('[Question', '').replace(']', '')
                relevant_answers = []

                # Look for relevant answers in all chunks
                for result in all_results:
                    for line in result.split('\n'):
                        if marker in line:
                            answer = line.replace(marker, "").strip()
                            answer = answer.replace(question_text, "").strip()
                            if answer and answer not in relevant_answers:
                                relevant_answers.append(answer)

                # Combine answers or provide default
                combined_answer = "\n".join(relevant_answers) if relevant_answers else "No relevant information found."

                formatted_output["questions"].append({
                    "question_number": question_number,
                    "question": question_text,
                    "answer": combined_answer
                })

            return formatted_output

        except Exception as e:
            logging.error(f"Error formatting response: {e}")
            return {
                "error": str(e),
                "raw_output": str(all_results)
            }

    def analysis(self, file_paths: List[str]) -> List[Dict]:
        """Compatible with original interface - processes multiple files."""
        results = []
        for file_path in file_paths:
            data = self.load_data(file_path)
            if not data:
                logging.error(f"No data loaded for {file_path}.")
                continue
            analysis_results = self.ask_questions(data)
            results.append({"file_path": file_path, "result": analysis_results})
        return results