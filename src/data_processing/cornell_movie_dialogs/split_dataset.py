import json
from pathlib import Path


def split_conversations(input_file, output_dir='split_conversations'):
    """Split a single JSON file containing multiple conversations into separate files"""
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    try:
        # Read the input JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            conversations = json.load(f)

        print(f"Found {len(conversations)} conversations to process")

        # Process each conversation
        for conversation in conversations:
            # Get conversation ID
            conv_id = conversation['original_id']

            # Create a new dictionary without the questions
            cleaned_conversation = {
                'original_id': conversation['original_id'],
                'dialogue': conversation['dialogue']
            }

            # Create output file path
            output_file = output_path / f"{conv_id}.json"

            # Write individual JSON file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(cleaned_conversation, f, indent=2, ensure_ascii=False)

            print(f"Created: {output_file}")

        print(f"\nSuccessfully created {len(conversations)} conversation files in {output_dir}/")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())


def main():
    # Set paths
    current_dir = Path.cwd()
    input_file = current_dir / 'formatted_conversations/formatted_conversations_1-1000.json'
    output_dir = current_dir / 'formatted_conversations/split'

    print(f"Looking for input file: {input_file}")
    print(f"Output directory will be: {output_dir}")

    # Split the conversations
    split_conversations(input_file, output_dir)


if __name__ == "__main__":
    main()