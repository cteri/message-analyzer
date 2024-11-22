import csv
import os
import random
from datetime import datetime, timedelta

import openai
from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# List of grooming-related questions
questions = [
    "Has anyone ever discussed their age in a chat?",
    "Have people talked about meeting in person?",
    "Has anyone given gifts to others?",
    "Has anyone ever asked for personal information?",
    "Have any conversations involved secret keeping?",
    "Has anyone expressed loneliness or a desire for companionship?",
    "Has anyone used flattery or compliments?",
    "Have any videos or photos been produced? requested?",
]


# Function to generate responses for question
def generate_responses(question, num_responses=20):
    responses = []

    for _ in range(num_responses):
        # Defining the system message to set the context
        system_message = {
            "role": "system",
            "content": "You are a chatbot that provides concise and straightforward responses.",
        }

        # Defining the user message with the question
        user_message = {
            "role": "user",
            "content": f"Generate a conversation related to the question: '{question}' without any introductions or extra context, where one participant (an adult) is a bit predatory or pushy, and the other participant (a child aged 10-13) is sometimes curious, cautious, witty and sometimes naive. Focus on realistic dialogue with emotional undertones.",
        }

        response = client.chat.completions.create(
            model="gpt-4o", messages=[system_message, user_message], max_tokens=400
        )

        # Print the entire API response for debugging
        print("API Response:", response)

        try:
            answer = response.choices[0].message.content.strip()
            responses.append(answer)
        except (IndexError, AttributeError) as e:
            print(f"Error extracting response: {e}")
            responses.append("Error generating response")

    return responses


# Main function to generate responses for all questions
def main():
    all_responses = []

    for question in questions:
        responses = generate_responses(question)
        for response in responses:
            all_responses.append({"question": question, "response": response})

    # Save the generated responses to a CSV file
    with open("eight_question_dataset.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "response"])
        writer.writeheader()
        writer.writerows(all_responses)

    print("Responses generated and saved to 'eight_question_dataset.csv'.")


if __name__ == "__main__":
    main()


# # Function to generate synthetic chat data
# def generate_chat_data(question, num_scenarios=20):
#     chat_logs = []

#     for _ in range(num_scenarios):
#         # Define the prompt for generating the chat
#         prompt = [
#             {"role": "system", "content": (
#                 f"Generate a realistic chat between two users where User1 is a groomer and User2 is a pre-teen. "
#                 f"User1 is trying to manipulate User2, who is 12 years old. "
#                 f"User2 is curious but cautious. "
#                 f"The chat should consist of at least 7-8 lines related to the question: '{question}'. "
#                 f"Include at least 4-5 lines of off-topic or irrelevant conversation to reflect realistic chat dynamics. "
#                 f"Include timestamps in the format 'YYYY-MM-DD HH:MM:SS'. "
#                 f"Make sure to separate the dialogue of User1 and User2 clearly."
#             )}
#         ]

#         # Call the OpenAI API to generate the chat
#         response = client.chat.completions.create(
#             model="gpt-4",  # Use the correct model name you have access to
#             messages=prompt,
#             max_tokens=800
#         )

#         # Extract and process the chat log
#         chat_log = response.choices[0].message.content.strip()

#         # Generate a random timestamp for each scenario
#         timestamp = datetime.now() - timedelta(
#             days=random.randint(0, 30),
#             hours=random.randint(0, 23),
#             minutes=random.randint(0, 59)
#         )
#         timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')

#         # Initialize dialogues
#         user1_dialogue = []
#         user2_dialogue = []

#         # Parse the chat log into User1 and User2 dialogues
#         for line in chat_log.split("\n"):
#             if line.startswith("User1:"):
#                 user1_dialogue.append(line.replace("User1: ", "").strip())
#             elif line.startswith("User2:"):
#                 user2_dialogue.append(line.replace("User2: ", "").strip())

#         # Append the scenario to the results
#         chat_logs.append({
#             "question": question,
#             "timestamp": timestamp_str,
#             "User1": ' '.join(user1_dialogue),
#             "User2": ' '.join(user2_dialogue)
#         })

#     return chat_logs


# # List of questions to generate scenarios for
# questions = [
#     "Has anyone given their age?",
#     "Has anyone asked to meet up in person? where?",
#     "Has anyone given a gift to the other?",
#     "Has anyone bought something from a list like an amazon wish list?",
#     "Have any videos or photos been produced or requested?",
#     "Has anyone offered any financial help?",
#     "Has anyone expressed feelings of loneliness or a desire for companionship?",
#     "Has anyone asked the other to keep a secret?",
#     "Has anyone discussed their location?",
#     "Has anyone expressed fear or discomfort in the conversation?",
#     "Has anyone used flattery or compliments?"

# ]

# # Main function to generate chat data for all questions
# def main():
#     all_chat_data = []

#     for question in questions:
#         chat_data = generate_chat_data(question, num_scenarios=20)  # Generate 20 scenarios
#         all_chat_data.extend(chat_data)

#     # Save the generated chat data to a CSV file
#     with open('synthetic_chat_data.csv', 'w', newline='', encoding='utf-8') as f:
#         writer = csv.DictWriter(f, fieldnames=["question", "timestamp", "User1", "User2"])
#         writer.writeheader()
#         writer.writerows(all_chat_data)

#     print("Synthetic chat data generated and saved to 'synthetic_chat_data.csv'.")

# if __name__ == "__main__":
#     main()
