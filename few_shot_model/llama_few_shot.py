import subprocess
import csv
import os

# Load your few-shot examples from the text file
def load_few_shot_examples(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file '{filename}' does not exist.")
    with open(filename, 'r') as file:
        examples = file.read().strip()
    return examples

# Load chat log contents in chunks
def load_chat_log_chunks(filename, chunk_size=10):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file '{filename}' does not exist.")
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        chat_log = [row for row in reader]
    
    # Break the chat log into chunks
    return [chat_log[i:i + chunk_size] for i in range(0, len(chat_log), chunk_size)]

# Format a single chat log chunk for the prompt
def format_chat_log_chunk(chunk):
    return "\n".join([f"{row[0]}: {row[1]} (Timestamp: {row[2]})" for row in chunk])

# Main loop to ask questions
def ask_questions():
    # Ask for the chat log file once
    chat_log_file = input("Please provide the chat log CSV filename: ")
    chat_log_chunks = load_chat_log_chunks(chat_log_file)
    
    # Load few-shot examples once
    few_shot_examples = load_few_shot_examples("few_shot_examples.txt")

    while True:
        # Ask for the question
        question = input("\nEnter your question: ")

        all_responses = []

        for i, chunk in enumerate(chat_log_chunks):
            chat_log_text = format_chat_log_chunk(chunk)
            
            # Concatenate few-shot examples, question, and chat log chunk for the prompt
            prompt = (
                f"Here are some few-shot examples of similar questions and their answers:\n\n"
                f"{few_shot_examples}\n\n"
                f"Now, examine this part of the chat log carefully and answer the question by "
                f"identifying specific details with timestamps and quoting relevant lines if available.\n\n"
                f"Chat Log Section {i+1}:\n{chat_log_text}\n\n"
                f"Question: {question}\n\n"
                f"Please provide a concise answer, including timestamps and direct quotes from the chat log where applicable."
            )

            # Run the prompt using `ollama`
            result = subprocess.run(
                ["ollama", "run", "llama3.1", prompt],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                response = result.stdout.strip()
                if response:  # Only add non-empty responses
                    all_responses.append(response)
            else:
                print("Error running the model:", result.stderr.strip())
                break

        # Display the aggregated answer
        if all_responses:
            print("Answer:", " ".join(all_responses))
        else:
            print("No relevant information found in the chat log.")

        # Ask if the user wants to ask another question
        continue_prompt = input("\nDo you want to ask another question? (y/n): ").strip().lower()
        if continue_prompt != 'y':
            print("Exiting the question loop.")
            break

# Run the main function
if __name__ == "__main__":
    ask_questions()
