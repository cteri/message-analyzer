import os

from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.constants import DataTypes
from flask_ml.flask_ml_server.models import (ImageResult, ResponseModel,
                                             TextResult)

from ..ml.model import OllamaModel

model = OllamaModel()
server = MLServer(__name__)


@server.route("/analyzer", input_type=DataTypes.CUSTOM)
def analyzer(inputs: list[dict], parameters: dict):
    try:
        # Validate that the input contains a file path
        input_data = inputs[
            0
        ]  # Since you only process one file at a time, take the first input

        # csv_file = input_data['input']['file_path']
        csv_file = input_data.input["file_path"]

        # Check if the CSV file exists
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file {csv_file} not found.")

        # Use the model to analyze the file
        analysis_result = model.analysis([csv_file])[0]

        # Prepare the output file path
        output_file = csv_file.replace(".csv", "_output.csv")

        result_details = {
            "output_file": output_file,
            "success": analysis_result['result']['success'],
            "message": analysis_result['result']['message']
        }

        # Create the text result
        # todo: fixme to the correct type
        text_result = [
            ImageResult(
                file_path=f"{csv_file}",  # Use the file path as the ID
                result=result_details,
            )
        ]

        # Return the single result in a list, as `ResponseModel` expects a list of results
        response = ResponseModel(results=text_result)

    except ValueError as e:
        # Handle input validation errors
        response = ResponseModel(errors=[str(e)])
    except Exception as e:
        # Handle any other unexpected errors
        response = ResponseModel(errors=["An unexpected error occurred", str(e)])

    return response.get_response()


server.run()
