import csv
import os
import subprocess
from datetime import datetime, timedelta

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load Sentence Transformer model for retrieval
retriever_model = SentenceTransformer("all-MiniLM-L6-v2")


def load_few_shot_examples(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file '{filename}' does not exist.")
    with open(filename, "r") as file:
        examples = file.read().strip()
    return examples


def load_chat_log(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file '{filename}' does not exist.")
    with open(filename, "r") as file:
        reader = csv.reader(file)
        return [row for row in reader]


def retrieve_relevant_messages(chat_log, question, top_n=5):
    question_embedding = retriever_model.encode([question])[0]
    message_embeddings = retriever_model.encode([row[1] for row in chat_log])
    similarities = cosine_similarity([question_embedding], message_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_n:][::-1]

    results = []
    for idx in top_indices:
        row = chat_log[idx]
        results.append(f"{row[0]}: {row[1]} (Timestamp: {row[2]})")

    return "\n".join(results)


def generate_response(question, context, few_shot_examples):
    prompt = (
        f"Here are some few-shot examples of similar questions and their answers:\n\n"
        f"{few_shot_examples}\n\n"
        f"Now, examine this part of the chat log carefully and answer the question by "
        f"identifying specific details with timestamps and quoting relevant lines if available.\n\n"
        f"Relevant Chat Log Excerpts:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Please provide a concise answer, including timestamps and direct quotes from the chat log where applicable."
    )

    result = subprocess.run(
        ["ollama", "run", "llama3.1", prompt], capture_output=True, text=True
    )

    if result.returncode == 0:
        return result.stdout.strip()
    else:
        return f"Error running the model: {result.stderr.strip()}"


def main():
    print("Chat Log Analysis with RAG")
    print("==========================")

    chat_log_file = input("Please provide the chat log CSV filename: ")
    chat_log = load_chat_log(chat_log_file)

    few_shot_examples = load_few_shot_examples("few_shot_examples.txt")

    while True:
        question = input("\nEnter your question (or 'quit' to exit): ")

        if question.lower() == "quit":
            break

        relevant_messages = retrieve_relevant_messages(chat_log, question)
        response = generate_response(question, relevant_messages, few_shot_examples)

        print("\nRelevant messages:")
        print(relevant_messages)
        print("\nAI Response:")
        print(response)

        continue_prompt = (
            input("\nDo you want to ask another question? (y/n): ").strip().lower()
        )
        if continue_prompt != "y":
            print("Exiting the question loop.")
            break


if __name__ == "__main__":
    main()
