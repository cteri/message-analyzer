# message-analyzer

## Prepare Huggingface token
* download ollama - https://github.com/ollama/ollama
* download llama3.1 model - `ollama pull llama3.1`

## Installation
### 1. Installing requirements
```
python3 -m pip install -r requirements.txt
```

### 2. Flask-ML
* Starting the server
```
python3 -m src.backend.server
```

### 3. Client example
#### 3.1 Api
```
python3 src/client/client.py
```

#### 3.2 CLI
```
python3 -m src.client.cmd_client --model_name llama3.2:1b --input_directory ./test --output_file ./folder/output/analysis_results.csv
```


python script.py --input_file conversations.json --output_file results.csv --model llama2

python3 -m src.client.analysis_client --input_file ./src/data_processing/cornell_movie_dialogs/split_conversations/conversations_part_001.json --models=llama3.2:1b
