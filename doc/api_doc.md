# API Documentation: Analyzer Service

## URI

`POST {URL}/analyzer`

---

## Headers

- `Accept: application/json`
- `Content-Type: application/json`

---

## Request Body

The request body should be in JSON format and contain the following fields:

| Parameter         | Data Type   | Required | Description                                                  |
| ----------------- | ----------- | -------- | ------------------------------------------------------------ |
| inputs            | array       | Yes      | An array of input objects. Each object contains the file path to be analyzed. |
| input.file_path   | string      | Yes      | The file path of the CSV or txt file to be analyzed.                |
| data_type         | string      | Yes      | Specifies the type of data. Example: `CUSTOM`.               |

### Example Request:

```json
{
    "inputs": [
        {
            "input": {
                "file_path": "http://127.0.0.1:5000/test/mock_conversation_1.csv"
            }
        }
    ],
    "data_type": "CUSTOM"
}
```

## Response Body

The response will be a JSON object that contains the status of the analysis and the results, with a multi-layered structure for the `result`.

| Field               | Data Type   | Description                                                  |
| ------------------- | ----------- | ------------------------------------------------------------ |
| `status`            | string      | The status of the request. Expected values: `SUCCESS`, `FAILURE`. |
| `results`           | array       | An array of result objects. Each result contains details about the analysis outcome. |
| `result.output_file`| string      | The path to the output file that was generated.               |
| `result.success`    | boolean     | Indicates whether the analysis was successful (`true`) or failed (`false`). |
| `result.message`    | string      | A detailed message describing the result of the analysis.     |
| `file_path`         | string      | The file path of the input file that was analyzed.            |

### Example Response

```json
{
    "status": "SUCCESS",
    "results": [
        {
            "result": {
                "output_file": "http://127.0.0.1:5000/output/mock_conversation_1_output.csv",
                "success": true,
                "message": "Analysis completed successfully"
            },
            "file_path": "http://127.0.0.1:5000/test/mock_conversation_1.csv"
        }
    ]
}
```
