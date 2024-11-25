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
python3 -m src.client.cmd_client --output_directory ./folder/output --input_files ./test/mock_conversation_1.csv
```
