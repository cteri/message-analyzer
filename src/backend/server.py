# from typing import List, TypedDict
# from pydantic import BaseModel
# from flask_ml.flask_ml_server import MLServer, load_file_as_string
# from flask_ml.flask_ml_server.models import (
#     BatchFileInput,
#     InputSchema,
#     InputType,
#     ParameterSchema,
#     TaskSchema,
#     BatchFileResponse,
#     FileResponse,
#     FileType,
#     TextParameterDescriptor,
# )
# from ..ml.model import LlamaModel
# import os
# import tempfile
# import json
# from enum import Enum

# # Create JobStatus enum for tracking processing status
# class JobStatus(str, Enum):
#     PENDING = "PENDING"
#     RUNNING = "RUNNING"
#     COMPLETED = "COMPLETED"
#     FAILED = "FAILED"

# # Data models
# class QuestionAnalysis(BaseModel):
#     question_number: str
#     question: str
#     answer: str
#     evidence: str
#     instances: list = []

# class Analysis(BaseModel):
#     questions: List[QuestionAnalysis]

# class AnalysisOutput(BaseModel):
#     file_path: str
#     conversation_ids: List[str] = []
#     analysis: Analysis

# class ResultDetails(BaseModel):
#     output_file: str
#     success: bool
#     message: str
#     results: AnalysisOutput
#     status: str = JobStatus.COMPLETED
#     progress: float = 100.0

# class AnalysisResult(BaseModel):
#     result: ResultDetails
#     file_path: str
#     job_id: str = ""

# class AnalyzerResponse(BaseModel):
#     status: str
#     results: List[AnalysisResult]
#     job_status: JobStatus = JobStatus.COMPLETED

# class AnalyzerInputs(TypedDict):
#     inputs: BatchFileInput

# class AnalyzerParameters(TypedDict):
#     data_type: str

# def get_analyzer_task_schema():
#     return TaskSchema(
#         inputs=[
#             InputSchema(
#                 key="inputs",
#                 label="Input Files",
#                 inputType=InputType.BATCHFILE,
#                 file_types=[FileType.CSV, FileType.TEXT],
#             )
#         ],
#         parameters=[
#             ParameterSchema(
#                 key="data_type",
#                 label="Data Type",
#                 value=TextParameterDescriptor(
#                     name="data_type",
#                     description="Type of data being analyzed",
#                     default="CUSTOM"
#                 ),
#             )
#         ],
#         response_body=BatchFileResponse(
#             files=[
#                 FileResponse(
#                     file_type=FileType.CSV,
#                     path="",
#                     description="Analysis results for each input file",
#                 )
#             ]
#         ),
#     )

# # Initialize model and server
# model = LlamaModel()
# server = MLServer(__name__)

# @server.route(
#     "/analyzer",
#     order=0,
#     short_title="Analyze Messages",
#     task_schema_func=get_analyzer_task_schema,
# )
# def analyzer(inputs: AnalyzerInputs, parameters: AnalyzerParameters) -> AnalyzerResponse:
#     try:
#         # Get input files
#         input_files = inputs.get('inputs')
#         if not input_files or not input_files.files:
#             raise ValueError("No input files provided")

#         results = []
#         total_files = len(input_files.files)

#         for index, file_input in enumerate(input_files.files, 1):
#             try:
#                 # Calculate progress
#                 progress = (index / total_files) * 100
                
#                 # Get file path
#                 file_path = getattr(file_input, 'path', None)
#                 if not file_path:
#                     raise ValueError("Invalid file input - no path found")

#                 # Process file
#                 with open(file_path, 'rb') as f:
#                     content = f.read()

#                 # Create temp file for processing
#                 with tempfile.NamedTemporaryFile(delete=False, mode='wb', suffix='.csv') as temp_file:
#                     temp_file.write(content)
#                     temp_file_path = temp_file.name

#                 # Run analysis
#                 analysis_results = model.analysis([temp_file_path])
                
#                 if not analysis_results or not analysis_results[0].get('result'):
#                     raise ValueError("Analysis failed to produce results")

#                 analysis_result = analysis_results[0]['result']
                
#                 # Create result output
#                 output_file = f"{os.path.splitext(os.path.basename(file_path))[0]}_output.csv"
                
#                 result = AnalysisResult(
#                     result=ResultDetails(
#                         output_file=output_file,
#                         success=True,
#                         message="Analysis completed successfully",
#                         results=analysis_result,
#                         status=JobStatus.COMPLETED,
#                         progress=progress
#                     ),
#                     file_path=file_path
#                 )
                
#                 results.append(result)

#             except Exception as e:
#                 # Handle individual file processing errors
#                 print(f"Error processing file: {str(e)}")
                
#                 # Create empty analysis for error case
#                 empty_analysis = AnalysisOutput(
#                     file_path=file_path if 'file_path' in locals() else "",
#                     conversation_ids=[],
#                     analysis=Analysis(questions=[
#                         QuestionAnalysis(
#                             question_number=str(i+1),
#                             question=f"Question {i+1}",
#                             answer="NO",
#                             evidence="No evidence found in conversation",
#                             instances=[]
#                         ) for i in range(5)
#                     ])
#                 )

#                 error_result = AnalysisResult(
#                     result=ResultDetails(
#                         output_file="",
#                         success=False,
#                         message=str(e),
#                         results=empty_analysis,
#                         status=JobStatus.FAILED,
#                         progress=progress
#                     ),
#                     file_path=file_path if 'file_path' in locals() else ""
#                 )
                
#                 results.append(error_result)

#         # Create successful response
#         return AnalyzerResponse(
#             status="SUCCESS",
#             results=results,
#             job_status=JobStatus.COMPLETED
#         )

#     except Exception as e:
#         # Handle global processing errors
#         print(f"Global error in analyzer: {str(e)}")
        
#         # Create error response
#         empty_analysis = AnalysisOutput(
#             file_path="",
#             conversation_ids=[],
#             analysis=Analysis(questions=[
#                 QuestionAnalysis(
#                     question_number=str(i+1),
#                     question=f"Question {i+1}",
#                     answer="NO",
#                     evidence="No evidence found in conversation",
#                     instances=[]
#                 ) for i in range(5)
#             ])
#         )

#         error_result = AnalysisResult(
#             result=ResultDetails(
#                 output_file="",
#                 success=False,
#                 message=str(e),
#                 results=empty_analysis,
#                 status=JobStatus.FAILED,
#                 progress=0.0
#             ),
#             file_path=""
#         )

#         return AnalyzerResponse(
#             status="FAILURE",
#             results=[error_result],
#             job_status=JobStatus.FAILED
#         )

# # Add metadata about the app
# current_dir = os.path.dirname(os.path.abspath(__file__))
# app_info_path = os.path.join(current_dir, "app-info.md")

# server.add_app_metadata(
#     name="Message Analyzer",
#     author="UMass Rescue",
#     version="0.1.0",
#     info=load_file_as_string(app_info_path),
# )

# if __name__ == "__main__":
#     server.run()

from typing import List, TypedDict, Optional
from pydantic import BaseModel
from flask_ml.flask_ml_server import MLServer, load_file_as_string
from flask_ml.flask_ml_server.models import (
    BatchFileInput,
    InputSchema,
    InputType,
    ParameterSchema,
    TaskSchema,
    ResponseBody,
    BatchFileResponse,
    FileResponse,
    FileType,
    TextParameterDescriptor,
    MarkdownResponse,
)
from ..ml.model import LlamaModel
import os
import tempfile
import json

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
        parameters=[
            ParameterSchema(
                key="data_type",
                label="Data Type",
                value=TextParameterDescriptor(
                    name="data_type",
                    description="Type of data being analyzed",
                    default="CUSTOM"
                ),
            )
        ],
    )

@server.route(
    "/analyzer",
    order=0,
    short_title="Analyze Messages",
    task_schema_func=get_analyzer_task_schema,
)
def analyzer(inputs: AnalyzerInputs, parameters: AnalyzerParameters) -> ResponseBody:
    try:
        input_files = inputs.get('inputs')
        if not input_files or not input_files.files:
            return ResponseBody(
                root=MarkdownResponse(
                    title="Analysis Failed",
                    value="No input files provided"
                )
            )

        all_results = []
        file_responses = []
        
        for file_input in input_files.files:
            try:
                file_path = file_input.path
                
                # Process the file
                with open(file_path, 'rb') as f:
                    content = f.read()
                    
                with tempfile.NamedTemporaryFile(delete=False, mode='wb', suffix='.csv') as temp_file:
                    temp_file.write(content)
                    temp_file_path = temp_file.name

                # Run analysis
                analysis_results = model.analysis([temp_file_path])
                
                if analysis_results and analysis_results[0].get('result'):
                    result = analysis_results[0]['result']
                    
                    # Convert to Pydantic model
                    analysis_result = AnalysisResult(
                        file_path=result['file_path'],
                        conversation_ids=result.get('conversation_ids', []),
                        analysis=Analysis(
                            questions=[
                                Question(**q) for q in result['analysis']['questions']
                            ]
                        )
                    )
                    
                    # Create output file
                    output_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_analysis.csv"
                    output_dir = os.path.join(os.path.dirname(__file__), "output")
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, output_filename)
                    
                    # Save results to file
                    with open(output_path, 'w') as f:
                        json.dump(analysis_result.model_dump(), f, indent=2)
                    
                    # Add to results
                    all_results.append(analysis_result)
                    file_responses.append(
                        FileResponse(
                            title=output_filename,
                            path=output_path,
                            file_type=FileType.CSV,
                            description=f"Analysis results for {os.path.basename(file_path)}"
                        )
                    )

                # Cleanup
                os.unlink(temp_file_path)

            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")

        if not file_responses:
            return ResponseBody(
                root=MarkdownResponse(
                    title="Analysis Failed",
                    value="No analysis results were generated"
                )
            )

        # Create a summary of the analysis
        summary = "## Analysis Results\n\n"
        for result in all_results:
            file_name = os.path.basename(result.file_path)
            summary += f"### {file_name}\n\n"
            
            summary += "| Question | Answer | Evidence |\n"
            summary += "|----------|---------|----------|\n"
            
            for q in result.analysis.questions:
                summary += f"| {q.question} | {q.answer} | {q.evidence} |\n"
            
            summary += "\n"

        # Return properly structured response
        return ResponseBody(
            root=BatchFileResponse(
                files=file_responses,
                markdown=MarkdownResponse(
                    title="Conversation Analysis Results",
                    value=summary
                )
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