from datasets import load_dataset
import pandas as pd

# Step 1: Load the dataset from HuggingFace
dataset = load_dataset("naver-clova-ix/cord-v2")

# Step 2: Specify the dataset split (e.g., "train", "validation", "test")
# You can choose the split you want. For demonstration, let's use "train".
data_split = dataset['train']

# Step 3: Convert the dataset split to a pandas DataFrame
df = pd.DataFrame(data_split)

# Step 4: Save the DataFrame as a CSV file
csv_file = "cord_v2_train.csv"
df.to_csv(csv_file, index=False)

print(f"Dataset successfully saved as {csv_file}")

