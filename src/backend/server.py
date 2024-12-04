import json
import os
import pandas as pd
import tempfile
import logging
from typing import List, Optional, TypedDict
from ..ml.prompt_ollama import get_all_answers

from flask_ml.flask_ml_server import MLServer, load_file_as_string
from flask_ml.flask_ml_server.models import (BatchFileInput, BatchFileResponse,
                                             FileResponse, FileType,
                                             InputSchema, InputType,
                                             MarkdownResponse, ParameterSchema,
                                             ResponseBody, TaskSchema,
                                             TextParameterDescriptor)
from pydantic import BaseModel

from ..ml.model import LlamaModel


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
    data_type: str

model = LlamaModel()
server = MLServer(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "folder", "output")

def get_analyzer_task_schema():
    return TaskSchema(
        inputs=[
            InputSchema(
                key="inputs",
                label="Input Files",
                # label="Input Folder",
                inputType=InputType.BATCHFILE,
                # inputType=InputType.DIRECTORY,
                file_types=[FileType.CSV, FileType.TEXT],
            )
        ],
        parameters=[
            ParameterSchema(
                key="data_type",
                # key="max_files",
                label="Data Type",
                # label="Maximum Files to Process",
                value=TextParameterDescriptor(
                    name="data_type",
                    description="Type of data being analyzed",
                    default="CUSTOM",
                ),
            )
        ],
    )

# @server.route("/analyzer", order=0, short_title="Analyze Messages", task_schema_func=get_analyzer_task_schema)
# def analyzer(inputs: AnalyzerInputs, parameters: AnalyzerParameters) -> ResponseBody:
#    try:
#        input_files = inputs.get("inputs")
#        if not input_files or not input_files.files:
#            return ResponseBody(root=MarkdownResponse(
#                title="Analysis Failed", 
#                value="No input files provided"
#            ))

#        all_results = []
#        file_responses = []

#        for file_input in input_files.files:
#            try:
#                file_path = file_input.path
#                output_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_analysis.json"

#                with open(file_path, "rb") as f:
#                    content = f.read()

#                with tempfile.NamedTemporaryFile(delete=False, mode="wb", suffix=".csv") as temp_file:
#                    temp_file.write(content)
#                    temp_file_path = temp_file.name

#                analysis_results = model.analysis([temp_file_path])

#                if analysis_results and analysis_results[0].get("result"):
#                    result = analysis_results[0]["result"]

#                    analysis_data = {
#                        "file_path": result["file_path"],
#                        "conversation_ids": result.get("conversation_ids", []),
#                        "analysis": {"questions": []}
#                    }
#                    analysis_data["file_path"] = os.path.join("folder", "output", output_filename)

#                    if "analysis" in result and "questions" in result["analysis"]:
#                        for q in result["analysis"]["questions"]:
#                            question_data = {
#                                "question_number": q["question_number"],
#                                "question": q["question"],
#                                "answer": q["answer"],
#                                "evidence": q["evidence"],
#                                "instances": q.get("instances", [])
#                            }
#                            analysis_data["analysis"]["questions"].append(question_data)

#                    # Consolidate file writing
#                    os.makedirs(OUTPUT_DIR, exist_ok=True)

#                     # Write JSON analysis
#                    output_path = os.path.join(OUTPUT_DIR, output_filename)
#                    with open(output_path, 'w', encoding='utf-8') as f:
#                         json.dump(analysis_data, f, indent=2)

#                     # Write highlighted CSV
#                    evidence_list = [q["evidence"] for q in result["analysis"]["questions"] if q["answer"] == "YES"]
#                 #    csv_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(output_filename)[0]}_highlighted.csv")
#                 #    highlighted_content = create_highlighted_csv(content, evidence_list)

#                 #    csv_content = content.decode('utf-8') if isinstance(content, bytes) else content
#                    highlighted_csv = create_highlighted_csv(content, evidence_list)
#                    csv_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(output_filename)[0]}_highlighted.csv")
#                    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
#                         f.write(highlighted_csv)

#                     # Modified file response
#                    file_responses.extend([
#                         FileResponse(
#                             title=output_filename,
#                             path=output_path,
#                             file_type=FileType.JSON,
#                             description="Analysis results"
#                         ),
#                         FileResponse(
#                             title=f"{os.path.splitext(output_filename)[0]}_highlighted.csv",
#                             path=csv_path,
#                             file_type=FileType.CSV,
#                             description="Highlighted conversation",
#                             properties={
#                                 "highlightedRows": [q["evidence"] for q in result["analysis"]["questions"] if q["answer"] == "YES"],
#                                 "highlightColor": "#FFEB3B"
#                             }
#                         )
#                     ])

#                    all_results.append(analysis_data)

#                os.unlink(temp_file_path)

#            except Exception as e:
#                logging.error(f"Error processing file {file_path}: {str(e)}")
#                error_analysis = {
#                    "file_path": os.path.join("folder", "output", output_filename),
#                    "conversation_ids": [],
#                    "analysis": {
#                        "questions": [
#                            {
#                                "question_number": str(i + 1),
#                                "question": f"Question {i + 1}",
#                                "answer": "NO",
#                                "evidence": "Error processing file",
#                                "instances": []
#                            } for i in range(5)
#                        ]
#                    }
#                }
#                all_results.append(error_analysis)

#        if not file_responses:
#            return ResponseBody(root=MarkdownResponse(
#                title="Analysis Failed", 
#                value="No analysis results were generated"
#            ))

#        summary = "## Analysis Results\n\n"
#        for result in all_results:
#            file_name = os.path.basename(result["file_path"])
#            summary += f"### {file_name}\n\n"
#            summary += "| Question | Answer | Evidence |\n"
#            summary += "|----------|---------|----------|\n"
           
#            for q in result["analysis"]["questions"]:
#                question = q["question"].replace("|", "\\|")
#                answer = q["answer"].replace("|", "\\|")
#                evidence = q["evidence"].replace("|", "\\|")
#                summary += f"| {question} | {answer} | {evidence} |\n"
           
#            summary += "\n"

#        return ResponseBody(
#            root=BatchFileResponse(
#                files=file_responses,
#                markdown=MarkdownResponse(
#                    title="Conversation Analysis Results", 
#                    value=summary
#                ),
#            )
#        )

#    except Exception as e:
#        logging.error(f"Global error in analyzer: {str(e)}")
#        return ResponseBody(
#            root=MarkdownResponse(
#                title="Analysis Failed",
#                value=f"Error during analysis: {str(e)}"
#            )
#        )
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
                with open(file_input.path, 'r') as f:
                    df = pd.read_csv(file_input.path)
                
                # Format conversation for prompt_ollama
                conversation = {
                    'turns': [
                        {'speaker': row['Speaker'], 'text': row['Message']} 
                        for _, row in df.iterrows()
                    ]
                }

                # Get answers using new prompt_ollama
                results = get_all_answers(conversation, "llama3.1")
                # Add debug printing
                print("\nDEBUG - Raw results structure:", results)
                for qid, result_data in results.items():
                    print(f"Processing{qid}: answer={result_data['answer']}, Evidence: {result_data['evidence']}")

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
                    "Q5": "🟣"   # Media
                }
                questions_map = {
    "Q1": "Has any person given their age? (and what age was given)",
    "Q2": "Has any person asked the other for their age?",
    "Q3": "Has any person asked to meet up in person? Where?",
    "Q4": "Has any person given a gift to the other?",
    "Q5": "Have any videos or photos been produced? Requested?"
}

                for qid, result_data in results.items():
                    question = questions_map[qid]
                    emoji = emoji_map.get(qid, "")
                    answer = result_data['answer']
                    evidence = result_data['evidence']
                    markdown_content += f"| {emoji} {question} | {answer} | {evidence} |\n"

                markdown_content += "\n### Full Conversation\n"
                markdown_content += "| Time | Speaker | Message | Matches |\n"
                markdown_content += "|------|---------|---------|----------|\n"

                # Add each message with any matches
                for _, row in df.iterrows():
                    matches = []
                    message_text = row['Message']
                    for qid, result_data in results.items():
                        if result_data['answer'] == "YES":
                            evidence_text = result_data['evidence']
                            # Remove speaker attribution like "Alice:" or "Bob:"
                            if ':' in evidence_text:
                                evidence_text = evidence_text.split(':', 1)[1]
                            # Remove quotes
                            evidence_text = evidence_text.replace('"', '').strip()
                            
                            if evidence_text in message_text:
                                matches.append(emoji_map[qid])
                            # Also check if message appears in evidence (handles partial matches)
                            elif message_text in evidence_text:
                                matches.append(emoji_map[qid])
                    
                    match_indicators = " ".join(matches) if matches else ""
                    markdown_content += f"| {row['Timestamp']} | {row['Speaker']} | {message_text} | {match_indicators} |\n"
                # for _, row in df.iterrows():
                #     matches = []
                #     message_text = row['Message']
                #     for qid, result_data in results.items():
                #         if result_data['answer'] == "YES":
                #             # Check if any part of the evidence appears in this message
                #             evidence_text = result_data['evidence']
                #             # Strip out any speaker attribution and parenthetical notes
                #             clean_evidence = evidence_text.split('"')[1] if '"' in evidence_text else evidence_text
                #             clean_evidence = clean_evidence.split('(')[0].strip()
                            
                #             if clean_evidence in message_text:
                #                 matches.append(emoji_map[qid])
                    
                #     match_indicators = " ".join(matches) if matches else ""
                #     markdown_content += f"| {row['Timestamp']} | {row['Speaker']} | {message_text} | {match_indicators} |\n"

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
                    title="Analysis Failed", 
                    value="No analysis results were generated"
                )
            )

        # Combine all markdown content
        final_markdown = "\n\n".join(all_results)

        return ResponseBody(
            root=MarkdownResponse(
                title="Conversation Analysis Results",
                value=final_markdown
            )
        )

    except Exception as e:
        print(f"Global error in analyzer: {str(e)}")
        return ResponseBody(
            root=MarkdownResponse(
                title="Analysis Failed",
                value=f"Error during analysis: {str(e)}"
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