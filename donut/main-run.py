from donut import DonutModel
from PIL import Image
import torch
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True, help='Model path')
parser.add_argument('--img_path', required=True, help='Image path')
args = parser.parse_args()

# Load model
model_path = args.model_path
img_path = args.img_path
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