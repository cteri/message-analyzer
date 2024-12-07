import argparse
import json

import pandas as pd

from ..ml.prompt_ollama import get_all_answers_for_conversations
parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, required=True)
parser.add_argument('--model', type=str, default='llama3.1')
parser.add_argument('--output_file', type=str, required=True)
args = parser.parse_args()

input_file = args.input_file
model = args.model
output_file = args.output_file

with open(input_file) as f:
    data = json.load(f)

answers = get_all_answers_for_conversations(data, model)

df = pd.DataFrame(answers)
df.to_csv(output_file, index=False)