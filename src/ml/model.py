import random


class OllamaModel:
    def analysis(self, file_paths: list[str]) -> list[dict]:
        # Analyze the files, returning a single result for each file
        return [
            {
                "file_path": file_path,
                "result": {
                    "success": random.choice([True, False]),
                    "message": (
                        "Analysis completed successfully"
                        if random.choice([True, False])
                        else "Analysis failed"
                    ),
                },
            }
            for file_path in file_paths
        ]
