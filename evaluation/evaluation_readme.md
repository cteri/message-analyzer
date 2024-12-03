# Evaluation Process Documentation
* This document describes the process for running model evaluations on conversation datasets using the provided scripts.


## Prerequisites
* download ollama - https://github.com/ollama/ollama
* download llama3.1 model - `ollama pull llama3.1`
* Installing requirements
```
python3 -m pip install -r requirements.txt
```

## Directory Structure
```
.
├── src/
│   ├── client/
│   │   └── csv_analysis_client.py (run the model)
│   └── data_processing/
│       └── cornell_movie_dialogs/
│           ├── split_conversations/
│           │   └── conversations_part_*.json (dialogs)
│           └── labeled_csv/
│               └── labeled_data_1-1000.csv (label)
└── evaluation/
    ├── report.py (compare result)
    └── conversations_part_*.csv (model response)
```


## Running the Evaluation
### 1. Process Individual Conversation Parts
* Run the analysis client on each conversation part using different model versions. Each command processes a specific portion of the dataset and generates a corresponding CSV output.

```
python3 -m src.client.csv_analysis_client --input_file ./src/data_processing/cornell_movie_dialogs/split_conversations/conversations_part_001.json --output_file conversations_part_001.csv --model=llama3.1  
python3 -m src.client.csv_analysis_client --input_file ./src/data_processing/cornell_movie_dialogs/split_conversations/conversations_part_002.json --output_file conversations_part_002.csv --model=llama3.1  
python3 -m src.client.csv_analysis_client --input_file ./src/data_processing/cornell_movie_dialogs/split_conversations/conversations_part_003.json --output_file conversations_part_003.csv --model=llama3.1  
python3 -m src.client.csv_analysis_client --input_file ./src/data_processing/cornell_movie_dialogs/split_conversations/conversations_part_004.json --output_file conversations_part_004.csv --model=llama3.1 
python3 -m src.client.csv_analysis_client --input_file ./src/data_processing/cornell_movie_dialogs/split_conversations/conversations_part_005.json --output_file conversations_part_005.csv --model=llama3.1 
python3 -m src.client.csv_analysis_client --input_file ./src/data_processing/cornell_movie_dialogs/split_conversations/conversations_part_006.json --output_file conversations_part_006.csv --model=llama3.1 
python3 -m src.client.csv_analysis_client --input_file ./src/data_processing/cornell_movie_dialogs/split_conversations/conversations_part_007.json --output_file conversations_part_007.csv --model=llama3.1 
python3 -m src.client.csv_analysis_client --input_file ./src/data_processing/cornell_movie_dialogs/split_conversations/conversations_part_008.json --output_file conversations_part_008.csv --model=llama3.1
python3 -m src.client.csv_analysis_client --input_file ./src/data_processing/cornell_movie_dialogs/split_conversations/conversations_part_009.json --output_file conversations_part_009.csv --model=llama3.1
python3 -m src.client.csv_analysis_client --input_file ./src/data_processing/cornell_movie_dialogs/split_conversations/conversations_part_010.json --output_file conversations_part_010.csv --model=llama3.1
```

### 2. Generate Evaluation Report
* After processing all conversation parts, generate a comparison report between the model outputs and labeled data:
```
python3 evaluation/report.py --labeled-data "src/data_processing/cornell_movie_dialogs/labeled_csv/labeled_data_1-1000.csv" --conv-pattern "evaluation/conversations_part_*.csv" --output results.csv
```
