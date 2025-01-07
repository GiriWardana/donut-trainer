from datasets import load_dataset
import pandas as pd
import json

# Step 1: Load the dataset from HuggingFace
dataset = load_dataset("naver-clova-ix/cord-v2")

# Step 2: Specify the dataset split (e.g., "train", "validation", "test")
# You can choose the split you want. For demonstration, let's use "train".
data_split = dataset['train']

# Step 3: Convert the dataset split to a pandas DataFrame
df = pd.DataFrame(data_split)

# Step 4: Parse the JSON string in the 'ground_truth' column and extract the 'gt_parse' part
df['gt_parse'] = df['ground_truth'].apply(lambda x: json.loads(x)['gt_parse'])

# Step 5: Save the 'gt_parse' column as a CSV file
csv_file = "gt_parse.csv"
df['gt_parse'].to_csv(csv_file, index=False)

print(f"gt_parse data successfully saved as {csv_file}")
