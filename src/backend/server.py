import json
import os
import tempfile
import logging
from typing import List, Optional, TypedDict
# from .conversation_viewer import create_highlighted_conversation

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

# def create_highlighted_csv(self, content: bytes, evidence_list: List[str]) -> str:
#     """Creates CSV with highlighted rows based on evidence."""
#     lines = content.decode('utf-8').splitlines()
#     output = [lines[0]]  # Keep header

#     # Clean and prepare evidence for matching
#     cleaned_evidence = []
#     for evidence in evidence_list:
#         if "No evidence found" not in evidence:
#             # Remove speaker prefix if exists
#             if "," in evidence:
#                 evidence = evidence.split(",", 1)[1]
#             cleaned_evidence.append(evidence.strip('"\' ').lower())

#     # Process each line
#     for line in lines[1:]:
#         parts = line.split(',', 2)
#         if len(parts) < 3:
#             output.append(line)
#             continue

#         timestamp, speaker, message = parts
#         message = message.strip('"')
        
#         # Check if this line matches any evidence
#         should_highlight = any(ev in message.lower() for ev in cleaned_evidence)
        
#         # Add highlight column
#         output.append(f'{timestamp},{speaker},"{message}",{"yellow" if should_highlight else ""}')
    
#     return "\n".join(output)

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
        # input_folder = inputs.get("inputs")
        # if not input_folder or not input_folder.folder_path:
            return ResponseBody(
                root=MarkdownResponse(
                    title="Analysis Failed", value="No input files provided"
                )
            )

        all_results = []
        file_responses = []

        for file_input in input_files.files:
            try:
                file_path = file_input.path
                output_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_analysis.json"

                # Process the file
                with open(file_path, "rb") as f:
                    content = f.read()

                with tempfile.NamedTemporaryFile(
                    delete=False, mode="wb", suffix=".csv"
                ) as temp_file:
                    temp_file.write(content)
                    temp_file_path = temp_file.name

                # Run analysis
                analysis_results = model.analysis([temp_file_path])

                if analysis_results and analysis_results[0].get("result"):
                    result = analysis_results[0]["result"]

                    # Ensure proper JSON structure
                    analysis_data = {
                        "file_path": result["file_path"],
                        "conversation_ids": result.get("conversation_ids", []),
                        "analysis": {
                            "questions": []
                        }
                    }
                    analysis_data["file_path"] = os.path.join("folder", "output", output_filename)

                    # Process each question
                    if "analysis" in result and "questions" in result["analysis"]:
                        for q in result["analysis"]["questions"]:
                            question_data = {
                                "question_number": q["question_number"],
                                "question": q["question"],
                                "answer": q["answer"],
                                "evidence": q["evidence"],
                                "instances": q.get("instances", [])
                            }
                            analysis_data["analysis"]["questions"].append(question_data)

                    # Create output file with proper JSON formatting
                    # output_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_analysis.json"
                    # output_dir = os.path.join(os.path.dirname(__file__), "output")
                    # os.makedirs(output_dir, exist_ok=True)
                    # output_path = os.path.join(output_dir, output_filename)
                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    output_path = os.path.join(OUTPUT_DIR, output_filename)

                    # Write JSON with proper formatting
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(analysis_data, f, indent=2, ensure_ascii=False)

                    # Add to results
                    all_results.append(analysis_data)
                    file_responses.append(
                        FileResponse(
                            title=output_filename,
                            path=output_path,
                            file_type=FileType.JSON,  # Changed to JSON
                            description=f"Analysis results for {os.path.basename(file_path)}",
                        )
                    )

                # Cleanup
                os.unlink(temp_file_path)

            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
                # Create error response with proper JSON structure
                error_analysis = {
                    "file_path": os.path.join("folder", "output", output_filename),
                    "conversation_ids": [],
                    "analysis": {
                        "questions": [
                            {
                                "question_number": str(i + 1),
                                "question": f"Question {i + 1}",
                                "answer": "NO",
                                "evidence": "Error processing file",
                                "instances": []
                            } for i in range(5)
                        ]
                    }
                }
                all_results.append(error_analysis)

        if not file_responses:
            return ResponseBody(
                root=MarkdownResponse(
                    title="Analysis Failed", value="No analysis results were generated"
                )
            )

        # Create properly formatted summary
        summary = "## Analysis Results\n\n"
        for result in all_results:
            file_name = os.path.basename(result["file_path"])
            summary += f"### {file_name}\n\n"
            summary += "| Question | Answer | Evidence |\n"
            summary += "|----------|---------|----------|\n"
            
            for q in result["analysis"]["questions"]:
                # Escape any pipe characters in the text
                question = q["question"].replace("|", "\\|")
                answer = q["answer"].replace("|", "\\|")
                evidence = q["evidence"].replace("|", "\\|")
                summary += f"| {question} | {answer} | {evidence} |\n"
            
            summary += "\n"

        return ResponseBody(
            root=BatchFileResponse(
                files=file_responses,
                markdown=MarkdownResponse(
                    title="Conversation Analysis Results", value=summary
                ),
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