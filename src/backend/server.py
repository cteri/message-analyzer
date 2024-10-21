import os
from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.constants import DataTypes
from flask_ml.flask_ml_server.models import ImageResult, ResponseModel

from ..ml.model import LlamaModel

model = LlamaModel()
server = MLServer(__name__)


@server.route("/analyzer", input_type=DataTypes.CUSTOM)
def analyzer(inputs: list[dict], parameters: dict):
    # try:
        input_data = inputs[0]

        csv_file = input_data.input['file_path']

        if not csv_file.startswith(('http://', 'https://')):
            if not os.path.isabs(csv_file):
                # base_dir = os.path.dirname(os.path.abspath(__file__))
                # csv_file = os.path.join(base_dir, csv_file)
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
                csv_file = os.path.join(project_root, csv_file.lstrip('./'))
                csv_file = os.path.normpath(csv_file)
        print(csv_file)

        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file {csv_file} not found.")


        test = model.analysis([csv_file])
        print(test)

        analysis_result = model.analysis([csv_file])[0]

        result_details = {
            "output_file": csv_file.replace(".csv", "_output.csv"),
            "success": True,
            "message": "Analysis completed successfully",
            "results": analysis_result["result"]
        }

        text_result = [
            ImageResult(
                file_path=csv_file,
                result=result_details
            )
        ]

        response = ResponseModel(results=text_result)

    # except ValueError as e:
    #     response = ResponseModel(errors=[str(e)])
    # except Exception as e:
    #     response = ResponseModel(errors=["An unexpected error occurred", str(e)])
    #
    # return response.get_response()


server.run()
