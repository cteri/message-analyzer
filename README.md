# message-analyzer

## Download ollama
* download ollama - https://github.com/ollama/ollama
* download llama3.1 model - `ollama pull llama3.1`

## Installation
### 1. Installing requirements
```
pip3 install -r requirements.txt
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
python3 -m src.client.cmd_client --input_file ./src/data_processing/cornell_movie_dialogs/split_conversations/conversations_part_000.json --output_file ./analysis_results.csv --model=llama3.1 
```


