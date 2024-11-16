# message-analyzer

## Installation
### 0. Installing requirements
```
python3 -m pip install -r requirements.txt
```

### 1. Flask-ML
* Starting the server
```
python3 -m src.backend.server
```

### 2. Client example
#### 2.1 Api
```
python3 src/client/client.py
```

#### 2.2 CLI
```
python3 -m src.client.cmd_client --model_id=meta-llama/Llama-3.2-1B-Instruct --output_directory ./folder/output --input_files ./test/mock_conversation_1.csv
python3 -m src.client.cmd_client --model_id=meta-llama/Llama-3.1-8B-Instruct --output_directory ./folder/output --input_files ./test/mock_conversation_1.csv
```

#### 2.3 testing
```
python3 -m src.client.csv_analysis_client --input_file labeled_data_1-1000.csv --output_file analysis_results.json --batch_size 20
python3 -m src.client.csv_analysis_client --input_file src/data_processing/cornell_movie_dialogs/labeled_csv/labeled_data_1-20.csv --output_file analysis_results.json --batch_size 20

python3 -m src.client.dialogue_analyzer \
  --input_file src/data_processing/cornell_movie_dialogs/formatted_conversations/formatted_conversations_1-10.json \
  --output_file analysis_results.json \
  --batch_size 5
```

python3 -m src.client.dialogue_analyzer \
  --input_file src/data_processing/cornell_movie_dialogs/formatted_conversations/formatted_conversations_1-10.json \
  --output_dir / \
  --batch_size 5

python3 script.py --input_file conversations.json --output_dir output/ --batch_size 5 --model_id model_name