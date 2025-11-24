from flask import Flask, request, render_template_string, redirect, url_for
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake"
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = r"best_model.pth"
NUM_CLASSES = 10

single_img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

model = models.resnet50(weights=None)
in_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features, NUM_CLASSES)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(state_dict, strict=False)
model.to(DEVICE)
model.eval()

HTML_FORM = """
<!doctype html>
<title>Image Classification</title>
<h2>Upload an image for land use classification</h2>
<form method=post enctype=multipart/form-data action="/predict">
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
"""

HTML_RESULT = """
<!doctype html>
<title>Prediction Result</title>
<h2>Prediction: {{ class_name }}</h2>
<img src="data:image/png;base64,{{ img_data }}" alt="Uploaded image" width="300">
<br/><a href="/">Try another image</a>
"""

import base64

def transform_image(file_stream):
    img = Image.open(file_stream).convert('RGB')
    input_tensor = single_img_transform(img).unsqueeze(0).to(DEVICE)
    return img, input_tensor

def predict(input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
        pred_idx = pred.item()
    return pred_idx

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_FORM)

@app.route('/predict', methods=['POST'])
def predict_route():
    file = request.files['file']
    if not file:
        return redirect(url_for('index'))
    img, input_tensor = transform_image(file.stream)
    pred_idx = predict(input_tensor)
    class_name = CLASSES[pred_idx] if 0 <= pred_idx < len(CLASSES) else str(pred_idx)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return render_template_string(HTML_RESULT, class_name=class_name, img_data=img_str)

if __name__ == '__main__':
    app.run(debug=True)
