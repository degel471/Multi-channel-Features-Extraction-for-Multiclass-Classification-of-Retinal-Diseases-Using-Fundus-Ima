from flask import Flask, request, jsonify, render_template
import os
import wikipediaapi
import uuid
import json
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import torch
from torchvision import transforms
from vessels import *
from PIL import Image


wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='OcularDiseaseAnalyzer/1.0 (degelangvil471@gmail.com)'

)
WIKIPEDIA_TITLE_MAP = {
    "Glaucoma": "Glaucoma",
    "Diabetes": "Diabetic retinopathy",
    "Cataract": "Cataract",
    "AMD": "Age-related macular degeneration",
    "Hypertension": "Hypertensive retinopathy",
    "Myopia": "Myopia",
    "Normal": None  # Explicitly mark this as no info needed
}

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load precomputed results if available
try:
    with open('static/results_precomputed.json') as f:
        precomputed_results = json.load(f)
except FileNotFoundError:
    precomputed_results = {}

# Define class labels
CLASS_NAMES = ["Normal", "Diabetes", "Glaucoma", "Cataract", "AMD", "Hypertension", "Myopia"]

# Load the trained model
model = ODIRClassifier(n_classes=7)
model.load_state_dict(torch.load("models/vessels_10_2.pth"))
model.eval()

# Define image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Convert matplotlib figure to base64 string
def plot_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# Create visualization figure
def create_figure(image, title, cmap=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap=cmap) if cmap else ax.imshow(image)
    ax.set_title(title, fontsize=12)
    ax.axis('off')
    return fig

# Apply preprocessing steps
def process_and_visualize(image_path):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    r, g, b = cv2.split(image_np)
    rgb_clahe = cv2.merge([clahe.apply(r), clahe.apply(g), clahe.apply(b)])
    green_clahe = cv2.merge([b, clahe.apply(g), r])

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    clahe_gray = clahe.apply(gray)
    grey_clahe = np.stack([clahe_gray]*3, axis=-1)

    vessel_img = cv2.medianBlur(clahe.apply(g), 5)
    vessel_img = cv2.adaptiveThreshold(vessel_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY_INV, 15, 5)
    vessel_img = cv2.morphologyEx(vessel_img, cv2.MORPH_OPEN, np.ones((2,2), np.uint8), iterations=1)
    vessel_img = np.stack([vessel_img]*3, axis=-1)

    steps = {
        'original': plot_to_base64(create_figure(image_np, "Original Image")),
        'rgb_clahe': plot_to_base64(create_figure(rgb_clahe, "RGB CLAHE")),
        'green_clahe': plot_to_base64(create_figure(green_clahe, "Green CLAHE")),
        'gray_clahe': plot_to_base64(create_figure(grey_clahe, "Grayscale CLAHE")),
        'vessels': plot_to_base64(create_figure(vessel_img, "Blood Vessels", cmap='gray')),
    }
    return steps

@app.route('/')
def home():
    return render_template('frontend.html')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        steps = process_and_visualize(filepath)

        # if filename in precomputed_results:
        #     result = precomputed_results[filename]
        # else:
        # Run model prediction
        image = Image.open(filepath).convert("RGB")
        tensor = transform(image)
        input_tensor = tensor.unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor, input_tensor, input_tensor, input_tensor)
            probs = torch.sigmoid(output).numpy()[0]

        pred_dict = {label: float(p) for label, p in zip(CLASS_NAMES, probs)}
        top_idx = int(np.argmax(probs))
        top_label = CLASS_NAMES[top_idx]

        # If the prediction is "Normal", skip Wikipedia explanation
        if top_label == "Normal" or WIKIPEDIA_TITLE_MAP.get(top_label) is None:
            explanation = "You are healthy!"
        # elif top_label == "Glaucoma" or WIKIPEDIA_TITLE_MAP.get(top_label) is "Glaucoma (disease)":
        #     explanation = "Glaucoma is a group of eye diseases that can lead to damage of the optic nerve. The optic nerve transmits visual information from the eye to the brain. Glaucoma may cause vision loss if left untreated. It has been called the silent thief of sight because the loss of vision usually occurs slowly over a long period of time.[5] A major risk factor for glaucoma is increased pressure within the eye, known as intraocular pressure (IOP).[1] It is associated with old age, a family history of glaucoma, and certain medical conditions or the use of some medications."
        else:
            wiki_title = WIKIPEDIA_TITLE_MAP[top_label]
            page = wiki_wiki.page(wiki_title)
            explanation = page.summary[:500] if page.exists() else "No Wikipedia info available."



        result = {
            'results': pred_dict,
            'top_prediction': top_label,
            'explanation': explanation
        }

            # Save to memory + disk
            # precomputed_results[filename] = result
            # with open('static/results_precomputed.json', 'w') as f:
            #     json.dump(precomputed_results, f, indent=2)

        return jsonify({
            'original': filepath,
            'processing_steps': steps,
            'results': result['results'],
            'top_prediction': result['top_prediction'],
            'explanation': result['explanation']
        })

    except Exception as e:
        return jsonify({'error': str(e)})


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# @app.route('/predict_by_name', methods=['POST'])
# def predict_by_name():
#     try:
#         data = request.get_json()
#         filename = data.get('filename')
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

#         if not filename:
#             return jsonify({'error': 'No filename provided.'})

#         if not os.path.exists(filepath):
#             return jsonify({'error': f'Image file {filename} not found on server.'})

#         steps = process_and_visualize(filepath)

#         if filename in precomputed_results:
#             result = precomputed_results[filename]
#         else:
#             image = Image.open(filepath).convert("RGB")
#             tensor = transform(image)
#             input_tensor = tensor.unsqueeze(0)

#             with torch.no_grad():
#                 output = model(input_tensor, input_tensor, input_tensor, input_tensor)
#                 probs = torch.sigmoid(output).numpy()[0]

#             pred_dict = {label: float(p) for label, p in zip(CLASS_NAMES, probs)}
#             top_idx = int(np.argmax(probs))
#             top_label = CLASS_NAMES[top_idx]
#             explain = top_label + "eye disease"
#             try:
#                 explanation = wikipedia.summary(explain, sentences=2)
#             except:
#                 explanation = "No Wikipedia info available."

#             result = {
#                 'results': pred_dict,
#                 'top_prediction': top_label,
#                 'explanation': explanation
#             }

#             precomputed_results[filename] = result
#             with open('static/results_precomputed.json', 'w') as f:
#                 json.dump(precomputed_results, f, indent=2)

#         return jsonify({
#             'original': filepath,
#             'processing_steps': steps,
#             'results': result['results'],
#             'top_prediction': result['top_prediction'],
#             'explanation': result['explanation']
#         })

#     except Exception as e:
#         return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
