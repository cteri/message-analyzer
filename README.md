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
python3 -m src.client.cmd_client --output_directory ./folder/output --input_files ./test/mock_conversation_1.csv
```