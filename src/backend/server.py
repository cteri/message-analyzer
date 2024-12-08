import json
import logging
import os
import tempfile
from typing import List, Optional, TypedDict

import pandas as pd
from flask_ml.flask_ml_server import MLServer, load_file_as_string
from flask_ml.flask_ml_server.models import (BatchFileInput, BatchFileResponse,
                                             FileResponse, FileType,
                                             InputSchema, InputType,
                                             MarkdownResponse, ParameterSchema,
                                             ResponseBody, TaskSchema,
                                             TextParameterDescriptor)
from pydantic import BaseModel

from ..ml.model import LlamaModel
from ..ml.prompt_ollama import get_all_answers  # Changed to prompt_ollama1


# Pydantic models for response structure
class Question(BaseModel):
    question_number: str
    question: str
    answer: str
    evidence: str
    instances: List[dict] = []

class Analysis(BaseModel):
    questions: List[Question]

class AnalysisResult(BaseModel):
    file_path: str
    conversation_ids: List[str] = []
    analysis: Analysis

class AnalyzerResult(BaseModel):
    status: str
    message: str
    file_responses: List[FileResponse]
    markdown_content: str

class AnalyzerInputs(TypedDict):
    inputs: BatchFileInput

class AnalyzerParameters(TypedDict):
    pass


model = LlamaModel()
server = MLServer(__name__)

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "folder", "output"
)

def get_analyzer_task_schema():
    return TaskSchema(
        inputs=[
            InputSchema(
                key="inputs",
                label="Input Files",
                inputType=InputType.BATCHFILE,
                file_types=[FileType.CSV, FileType.TEXT],
            )
        ],
        parameters=[],
    )

@server.route(
    "/analyzer",
    order=0,
    short_title="Analyze Messages",
    task_schema_func=get_analyzer_task_schema,
)
def analyzer(inputs: AnalyzerInputs, parameters: AnalyzerParameters) -> ResponseBody:
    try:
        input_files = inputs.get("inputs")
        if not input_files or not input_files.files:
            return ResponseBody(
                root=MarkdownResponse(
                    title="Analysis Failed", value="No input files provided"
                )
            )

        all_results = []

        for file_input in input_files.files:
            try:
                # Read the CSV file
                with open(file_input.path, "r") as f:
                    df = pd.read_csv(file_input.path)

                # Format conversation for prompt_ollama
                conversation = {
                    "turns": [
                        {"speaker": row["Speaker"], "text": row["Message"]}
                        for _, row in df.iterrows()
                    ]
                }
                
                results, evidence_matches = get_all_answers(conversation, "llama3.1")
                
                # Add debug printing
                print("\nDEBUG - Raw results structure:", results)
                for qid, result_data in results.items():
                    print(
                        f"Processing{qid}: answer={result_data['answer']}, Evidence: {result_data['evidence']}"
                    )

                # Create markdown content
                markdown_content = f"""## Analysis Results for {os.path.basename(file_input.path)}

| Question | Answer | Evidence |
|----------|---------|----------|
"""
                emoji_map = {
                    "Q1": "🟡",  # Age given
                    "Q2": "🟠",  # Age asked
                    "Q3": "🟢",  # Meet up
                    "Q4": "🔵",  # Gift/Purchase
                    "Q5": "🟣",  # Media
                }
                questions_map = {
                    "Q1": "Has any person given their age? (and what age was given)",
                    "Q2": "Has any person asked the other for their age?",
                    "Q3": "Has any person asked to meet up in person? Where?",
                    "Q4": "Has any person given a gift to the other?",
                    "Q5": "Have any videos or photos been produced? Requested?",
                }

                for qid, result_data in results.items():
                    question = questions_map[qid]
                    emoji = emoji_map.get(qid, "")
                    answer = result_data["answer"]
                    evidence = result_data["evidence"]
                    markdown_content += f"| {emoji} {question} | {answer} | {evidence} |\n"

                markdown_content += "\n### Full Conversation\n"
                markdown_content += "| Time | Speaker | Message | Matches |\n"
                markdown_content += "|------|---------|---------|----------|\n"

                # Add each message with any matches
                for idx, row in df.iterrows():
                    matches = []
                    message_text = row["Message"]
                    
                    # Check each question's evidence lines for matches
                    for qid, result_data in results.items():
                        if result_data["answer"] == "YES":
                            if idx in result_data.get("evidence_lines", []):
                                matches.append(emoji_map[qid])

                    match_indicators = " ".join(matches) if matches else ""
                    markdown_content += f"| {row['Timestamp']} | {row['Speaker']} | {message_text} | {match_indicators} |\n"

                all_results.append(markdown_content)

            except Exception as e:
                print(f"Error processing file {file_input.path}: {str(e)}")
                error_content = f"""## Error Processing {os.path.basename(file_input.path)}
                
Error: {str(e)}
"""
                all_results.append(error_content)

        if not all_results:
            return ResponseBody(
                root=MarkdownResponse(
                    title="Analysis Failed", value="No analysis results were generated"
                )
            )

        # Combine all markdown content
        final_markdown = "\n\n".join(all_results)

        return ResponseBody(
            root=MarkdownResponse(
                title="Conversation Analysis Results", value=final_markdown
            )
        )

    except Exception as e:
        print(f"Global error in analyzer: {str(e)}")
        return ResponseBody(
            root=MarkdownResponse(
                title="Analysis Failed", value=f"Error during analysis: {str(e)}"
            )
        )

# Add metadata about the app
current_dir = os.path.dirname(os.path.abspath(__file__))
app_info_path = os.path.join(current_dir, "app-info.md")

server.add_app_metadata(
    name="Message Analyzer",
    author="UMass Rescue",
    version="0.1.0",
    info=load_file_as_string(app_info_path),
)

if __name__ == "__main__":
    server.run()
