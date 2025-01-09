from donut import DonutModel
from PIL import Image
import torch
import argparse
import json
import os

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True, help='Model path')
parser.add_argument('--img_path', required=True, help='Image path')
args = parser.parse_args()

# Load model
model_path = args.model_path
img_path = args.img_path
file_name = os.path.basename(img_path)
model = DonutModel.from_pretrained(model_path)

model = DonutModel.from_pretrained(model_path)
if torch.cuda.is_available():
    model.half()
    device = torch.device("cuda")
    model.to(device)
else:
    model.encoder.to(torch.bfloat16)

model.eval()

# Load image
image = Image.open(img_path).convert("RGB")

# Run inference
with torch.no_grad():
    output = model.inference(image=image, prompt="<s_train>")
    print(output)

    # Extracting first output
    first_output = output['predictions'][0]

    # Preparing second output
    second_output = {
        "file_name": file_name,
        "ground_truth": json.dumps({"gt_parse": first_output}, ensure_ascii=False)
    }

    # Printing outputs in JSONL format
    first_output_jsonl = json.dumps(first_output, ensure_ascii=False)
    second_output_jsonl = json.dumps(second_output, ensure_ascii=False)

    # print("🚀 ~ second_output_jsonl:", second_output_jsonl)
    # print("🚀 ~ first_output_jsonl:", first_output_jsonl)

    # File paths
    first_output_file = 'output_1.jsonl'
    second_output_file = 'output_2.jsonl'

    # Writing outputs to JSONL files
    with open(first_output_file, 'a') as f1:
        f1.write(first_output_jsonl + '\n')

    with open(second_output_file, 'a') as f2:
        f2.write(second_output_jsonl + '\n')

