# message-analyzer
* A tool for analyzing conversations using large language models.

## Prerequisites
### Ollama Setup
* Download and install Ollama from GitHub
  * https://github.com/ollama/ollama
* Pull the LLaMA 3.1 model:`ollama pull llama3.1`

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
* CLI Parameters
  * `--input_file`: Path to the input JSON file containing conversations
  * `--output_file`: Path where the analysis results will be saved (CSV format)
  * `--model`: Name of the LLM model to use (default: llama3.1)


### Project Structure
```
message-analyzer/
├── src/
│   ├── backend/        # Flask server implementation
│   ├── client/         # API and CLI clients
│   └── data_processing/# Data processing utilities
├── requirements.txt    # Python dependencies
└── README.md          # This file
```
