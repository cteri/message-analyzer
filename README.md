# message-analyzer

## Prepare Huggingface token
* Before switching to Ollama, please request authorization to access the Llama model using the link below. Note: The approval process may take some time.
* https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
* https://huggingface.co/meta-llama/Llama-3.1-8B
* After obtaining approval, please create a token in order to download the model.

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
