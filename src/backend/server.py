from typing import List, TypedDict
from pydantic import BaseModel
from flask_ml.flask_ml_server import MLServer, load_file_as_string
from flask_ml.flask_ml_server.models import (
    BatchDirectoryInput,
    BatchFileInput,
    EnumParameterDescriptor,
    EnumVal,
    InputSchema,
    InputType,
    ParameterSchema,
    ResponseBody,
    TaskSchema,
    MarkdownResponse,
    TextParameterDescriptor,
    BatchFileResponse,
    FileResponse,
    FileType,
)
from ..ml.model import LlamaModel
import os
import tempfile

class ResultDetails(BaseModel):
    output_file: str
    success: bool
    message: str
    results: list

class AnalysisResult(BaseModel):
    result: ResultDetails
    file_path: str

class AnalyzerResponse(BaseModel):
    status: str
    results: List[AnalysisResult]

class AnalyzerInputs(TypedDict):
    inputs: BatchFileInput

class AnalyzerParameters(TypedDict):
    data_type: str

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
        response_body=BatchFileResponse(
            files=[
                FileResponse(
                    file_type=FileType.CSV,
                    path="",
                    description="Analysis results for each input file",
                )
            ]
        ),
    )

model = LlamaModel()
server = MLServer(__name__)

@server.route(
    "/analyzer",
    order=0,
    short_title="Analyze Messages",
    task_schema_func=get_analyzer_task_schema,
)
def analyzer(inputs: AnalyzerInputs, parameters: AnalyzerParameters) -> AnalyzerResponse:
    try:
        # Extract files from input
        input_files = inputs.get('inputs')
        if not input_files or not input_files.files:
            raise ValueError("No input files provided")

        results = []
        
        for file_input in input_files.files:
            # Handle FileInput object
            if hasattr(file_input, 'read'):
                # Create a temporary file to store the content
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
                    temp_file.write(file_input.read())
                    csv_file = temp_file.name
            else:
                # Handle string path
                csv_file = str(file_input)
                if not os.path.isabs(csv_file):
                    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
                    csv_file = os.path.join(project_root, csv_file.lstrip('./'))
                    csv_file = os.path.normpath(csv_file)

            if not os.path.exists(csv_file):
                raise FileNotFoundError(f"File not found: {csv_file}")

            # Perform analysis
            analysis_result = model.analysis([csv_file])[0]

            # Create output filename
            output_file = os.path.splitext(os.path.basename(csv_file))[0] + "_output.csv"
            
            result_details = ResultDetails(
                output_file=output_file,
                success=True,
                message="Analysis completed successfully",
                results=analysis_result["result"]
            )

            results.append(AnalysisResult(
                result=result_details,
                file_path=csv_file
            ))

            # Clean up temporary file if created
            if hasattr(file_input, 'read'):
                os.unlink(csv_file)

        return AnalyzerResponse(
            status="SUCCESS",
            results=results
        )

    except Exception as e:
        return AnalyzerResponse(
            status="FAILURE",
            results=[
                AnalysisResult(
                    result=ResultDetails(
                        output_file="",
                        success=False,
                        message=str(e),
                        results=[]
                    ),
                    file_path="" if 'csv_file' not in locals() else csv_file
                )
            ]
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