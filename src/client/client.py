# from flask_ml.flask_ml_client import MLClient
# from flask_ml.flask_ml_server.constants import DataTypes

# url = "http://127.0.0.1:5000/analyzer"  # The URL of the server
# client = MLClient(url)  # Create an instance of the MLClient object

# inputs = [
#     {"input": {"file_path": "./test/mock_conversation_1.csv"}}
# ]  # The inputs to be sent to the server

# data_type = DataTypes.CUSTOM  # The type of the input data
# response = client.request(inputs=inputs, data_type=data_type)
# # response = client.request(inputs=inputs, parameters=parameters)

# print(response)  # Print the response

from flask_ml.flask_ml_client import MLClient

url = "http://127.0.0.1:5000/analyzer"  # The URL of the server
client = MLClient(url)  # Create an instance of the MLClient object

inputs = {"inputs": {"path": "./test"}}  # Specify the input folder path

parameters = {"max_files": 5}  # Specify the maximum number of files to process

response = client.request(inputs=inputs, parameters=parameters)

print(response)  # Print the response
