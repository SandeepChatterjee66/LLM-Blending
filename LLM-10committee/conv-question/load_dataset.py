import json
import os

# downloaded ConvQuestions dataset is in a directory inside /sandeep/datasets/ like this:
# conv_questions/
# ├── train.json
# ├── dev.json
# └── test.json

dataset_base_path = "../../sandeep/datasets/conv_questions" # IMPORTANT: Change this to your actual path

def load_convquestions_json(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f: # Assuming it might be JSON Lines (one JSON object per line)
                data.append(json.loads(line))
            # If it's a single JSON array:
            # data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}. Check file format.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

train_file = os.path.join(dataset_base_path, "train.json")
dev_file = os.path.join(dataset_base_path, "dev.json")
test_file = os.path.join(dataset_base_path, "test.json")

train_data = load_convquestions_json(train_file)
dev_data = load_convquestions_json(dev_file)
test_data = load_convquestions_json(test_file)

if train_data:
    print(f"Loaded {len(train_data)} training examples.")
    print("\nFirst training example:")
    print(json.dumps(train_data[0], indent=2))

if dev_data:
    print(f"Loaded {len(dev_data)} development examples.")

if test_data:
    print(f"Loaded {len(test_data)} test examples.")