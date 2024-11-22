import math
import os
import urllib.request
import zipfile
from typing import Dict, List

import pandas as pd


class CornellMovieDialogsCorpus:
    def __init__(
        self,
        data_dir: str = "cornell_movie_dialogs_corpus",
        output_dir: str = "split_conversations",
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.zip_path = "cornell_movie_dialogs_corpus.zip"
        self.download_url = (
            "http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"
        )

    def download_and_extract(self):
        """Download and extract the dataset"""
        os.makedirs(self.data_dir, exist_ok=True)

        if not os.path.exists(self.zip_path):
            print("Downloading Cornell Movie Dialogs Corpus...")
            urllib.request.urlretrieve(self.download_url, self.zip_path)
            print("Download completed!")

        if not os.path.exists(os.path.join(self.data_dir, "movie_lines.txt")):
            print("Extracting files...")
            with zipfile.ZipFile(self.zip_path, "r") as zip_ref:
                file_list = zip_ref.namelist()
                root_dir = file_list[0].split("/")[0]
                zip_ref.extractall()

                if root_dir != self.data_dir:
                    for file_name in os.listdir(root_dir):
                        old_path = os.path.join(root_dir, file_name)
                        new_path = os.path.join(self.data_dir, file_name)
                        os.rename(old_path, new_path)
                    os.rmdir(root_dir)
            print("Extraction completed!")

    def load_conversations(self) -> List[Dict]:
        """Load and process conversation data"""
        # Read conversation lines
        lines = {}
        lines_path = os.path.join(self.data_dir, "movie_lines.txt")
        print("Loading movie lines...")
        with open(lines_path, "r", encoding="iso-8859-1") as f:
            for line in f:
                parts = line.strip().split(" +++$+++ ")
                if len(parts) == 5:
                    lines[parts[0]] = {
                        "character_id": parts[1],
                        "movie_id": parts[2],
                        "character_name": parts[3],
                        "text": parts[4],
                    }

        # Read conversations
        conversations = []
        conv_path = os.path.join(self.data_dir, "movie_conversations.txt")
        print("Loading conversations...")
        with open(conv_path, "r", encoding="iso-8859-1") as f:
            for i, line in enumerate(f):
                parts = line.strip().split(" +++$+++ ")
                conversation_ids = eval(parts[3])
                turns = []
                for line_id in conversation_ids:
                    if line_id in lines:
                        turns.append(
                            {
                                "speaker": lines[line_id]["character_name"],
                                "text": lines[line_id]["text"],
                            }
                        )
                if turns:  # Only add non-empty conversations
                    conversations.append(
                        {
                            "conversation_id": f"conv_{i + 1}",
                            "movie_id": parts[2],
                            "turns": turns,
                        }
                    )

        return conversations

    def save_split_conversations(
        self, conversations: List[Dict], batch_size: int = 100
    ):
        """Save conversations in multiple files, each containing batch_size conversations"""
        import json

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Calculate number of files needed
        num_conversations = len(conversations)
        num_files = math.ceil(num_conversations / batch_size)

        print(
            f"\nSplitting {num_conversations} conversations into {num_files} files..."
        )

        # Split and save conversations
        for i in range(num_files):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_conversations)
            batch = conversations[start_idx:end_idx]

            # Create filename with padding zeros for correct sorting
            filename = os.path.join(
                self.output_dir, f"conversations_part_{i + 1:03d}.json"
            )

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(batch, f, ensure_ascii=False, indent=2)

            print(f"Saved {filename} with {len(batch)} conversations")


def main():
    # Initialize corpus processor
    corpus = CornellMovieDialogsCorpus()

    # Download and extract data
    corpus.download_and_extract()

    # Load conversations
    conversations = corpus.load_conversations()

    # Display basic statistics
    print(f"\nTotal number of conversations: {len(conversations)}")
    print("\nExample conversation:")
    example = conversations[0]
    print(f"Conversation ID: {example['conversation_id']}")
    print(f"Movie ID: {example['movie_id']}")
    for turn in example["turns"]:
        print(f"{turn['speaker']}: {turn['text']}")

    # Save split conversations
    corpus.save_split_conversations(conversations)

    print("\nProcessing completed!")


if __name__ == "__main__":
    main()
