import os
import torch
import transformers
import json
import logging

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
        # Set up model and tokenizer
        self.model_id = model_id
        self.pipeline = self.setup_model_and_tokenizer()

    def setup_model_and_tokenizer(self):
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
            tokenizer.pad_token_id = tokenizer.eos_token_id
            model = transformers.AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.float16,
                                                                      device_map="auto")
            return transformers.pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
        except Exception as e:
            logging.error(f"Error setting up model: {e}")
            return None

    def load_data(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif ext == '.txt' or ext == '.csv':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = [{"conversation_text": f.read().replace('\n', ' ')}]  # Read entire file and replace newlines
            else:
                logging.error(f"Unsupported file format: {ext}")
                return []
            return data
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            return []

    def ask_questions(self, conversation_data):
        try:
            prompt = f"""Please carefully read the following conversations and answer the questions. 
            Ensure your answers are accurate and well-supported.

            Conversations:
            {conversation_data}

            Please answer these questions:
            {questions}

            Please answer each question based on the conversation content. If no relevant information is found, 
            please explicitly state so.
            """
            output = self.pipeline(prompt, max_new_tokens=5000, do_sample=True, temperature=0.6, top_p=0.9)

            full_answer = output[0]['generated_text'].strip()

            formatted_answer = self.clean_and_format_response(prompt, full_answer, questions)

            return [{
                "conversation_ids": [conv.get('id', 'unknown') for conv in conversation_data],
                "analysis": formatted_answer
            }]
        except Exception as e:
            logging.error(f"Error in ask_questions: {e}")
            return [{
                "error": str(e)
            }]

    def clean_and_format_response(self, input_prompt: str, output_text: str, questions: str) -> dict:
        """
        Clean and format the model's response into JSON format

        Args:
            input_prompt: Original input prompt
            output_text: Raw output from the model
            questions: Original questions string

        Returns:
            Dictionary with formatted Q&A pairs
        """
        try:
            # Remove the input prompt from the output
            response = output_text.replace(input_prompt, '').strip()

            # Create a list of questions with their markers
            question_pairs = []
            for q in questions.split('\n'):
                if '[Question' in q:
                    marker = q[q.find('['):q.find(']') + 1]  # Extract [QuestionX]
                    question_text = q.replace(marker + '.', '').strip()
                    question_pairs.append((marker, question_text))

            # Split response into answers
            answers = []
            current_answer = ""
            for line in response.split('\n'):
                if '[Question' in line:
                    if current_answer:
                        answers.append(current_answer.strip())
                    current_answer = line
                elif line.strip():
                    current_answer += "\n" + line
            if current_answer:
                answers.append(current_answer.strip())

            # Create JSON structure
            formatted_output = {
                "questions": []
            }

            # Process each question and find its answer
            for marker, question_text in question_pairs:
                question_number = marker.replace('[Question', '').replace(']', '')
                answer = "No answer provided for this question."

                # Look for matching answer
                for ans in answers:
                    if marker in ans:
                        # Remove marker and question text from answer
                        answer = ans.replace(marker, "").strip()
                        answer = answer.replace(question_text, "").strip()
                        break

                # Add to JSON structure
                formatted_output["questions"].append({
                    "question_number": question_number,
                    "question": question_text,
                    "answer": answer
                })

            return formatted_output
        except Exception as e:
            logging.error(f"Error formatting response: {e}")
            return {
                "error": str(e),
                "raw_output": output_text
            }

    def analysis(self, file_paths: list[str]) -> list[dict]:
        results = []
        for file_path in file_paths:
            data = self.load_data(file_path)
            if not data:
                logging.error(f"No data loaded for {file_path}.")
                continue
            analysis_results = self.ask_questions(data)
            results.append({"file_path": file_path, "result": analysis_results})
        return results
