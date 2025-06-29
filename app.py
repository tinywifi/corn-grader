from flask import Flask, request, render_template_string, send_from_directory, jsonify
from PIL import Image, ImageDraw
from prometheus_flask_exporter import PrometheusMetrics
from PIL.ExifTags import TAGS
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from inference_sdk.http.errors import HTTPCallErrorError
import os, uuid, json, random, io, threading, time
import numpy as np
from flask_cors import CORS
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False

API_KEY = os.environ.get("ROBOFLOW_API_KEY")
PROJECT_ID = "corn-hub/4"
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
CLIENT = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=API_KEY)

CLIENT.configure(InferenceConfiguration(confidence_threshold=0.10))

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app = Flask(__name__)
metrics = PrometheusMetrics(app)
CORS(app)

progress_store = {}

USDA_GRADES = [
    ("U.S. No. 1", 3.0, 1, 0.1, 0),
    ("U.S. No. 2", 5.0, 2, 0.2, 0),
    ("U.S. No. 3", 7.0, 3, 0.5, 0),
    ("U.S. No. 4", 10.0, 5, 1.0, 0),
    ("U.S. No. 5", 15.0, 7, 3.0, 1)
]

def non_max_suppression(predictions, iou_threshold=0.3):
    boxes = []
    for pred in predictions:
        x0 = pred['x'] - pred['width'] / 2
        y0 = pred['y'] - pred['height'] / 2
        x1 = pred['x'] + pred['width'] / 2
        y1 = pred['y'] + pred['height'] / 2
        boxes.append((x0, y0, x1, y1, pred['confidence'], pred['class']))

    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    keep = []

    while boxes:
        chosen = boxes.pop(0)
        keep.append(chosen)
        # Remove boxes that overlap with chosen, regardless of class
        boxes = [box for box in boxes if iou(box, chosen) < iou_threshold]

    return [ {
        'x': (b[0]+b[2])/2,
        'y': (b[1]+b[3])/2,
        'width': b[2]-b[0],
        'height': b[3]-b[1],
        'confidence': b[4],
        'class': b[5]
    } for b in keep ]

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def classify_grade(damage_pct, damage_kernels, heat_pct, heat_kernels, total_kernels):
    if total_kernels < 100:
        for grade, _, max_damage_k, _, max_heat_k in USDA_GRADES:
            if damage_kernels <= max_damage_k and heat_kernels <= max_heat_k:
                return grade
    else:
        for grade, max_damage_pct, _, max_heat_pct, _ in USDA_GRADES:
            if damage_pct <= max_damage_pct and heat_pct <= max_heat_pct:
                return grade
    return "Sample Grade"

def compress_image(image_path, max_size_mb=5, quality=85, progress_id=None):
    def update_progress(step, total):
        progress = int((step / total) * 30) + 10
        if progress_id and progress_id in progress_store:
            progress_store[progress_id]['progress'] = progress
    
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        original_size = img.size
        
        update_progress(1, 5)
        
        target_pixels = 1920 * 1920
        current_pixels = img.size[0] * img.size[1]
        
        if current_pixels > target_pixels:
            ratio = (target_pixels / current_pixels) ** 0.5
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            update_progress(2, 5)
        
        for attempt in range(3):
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality, optimize=True)
            size_mb = buffer.tell() / (1024 * 1024)
            update_progress(3 + attempt, 5)
            
            if size_mb <= max_size_mb or quality <= 20:
                with open(image_path, 'wb') as f:
                    f.write(buffer.getvalue())
                return size_mb
            
            quality = max(20, quality - 20)

def process_image_async(filepath, filename, moisture, weight, progress_id):
    try:
        progress_store[progress_id] = {'progress': 0, 'status': 'processing', 'result': None, 'error': None}
        progress_store[progress_id]['progress'] = 5
        
        compress_image(filepath, progress_id=progress_id)
        progress_store[progress_id]['progress'] = 40
        
        try:
            raw_result = CLIENT.infer(filepath, model_id=PROJECT_ID)
            progress_store[progress_id]['progress'] = 70
        except HTTPCallErrorError as e:
            if e.status_code == 413:
                compress_image(filepath, max_size_mb=2, progress_id=progress_id)
                try:
                    raw_result = CLIENT.infer(filepath, model_id=PROJECT_ID)
                    progress_store[progress_id]['progress'] = 70
                except HTTPCallErrorError as retry_error:
                    if retry_error.status_code == 413:
                        progress_store[progress_id]['error'] = "Image too large after compression"
                        progress_store[progress_id]['status'] = 'error'
                        return
                    raise retry_error
            else:
                raise e

        result = raw_result.copy()
        result['predictions'] = non_max_suppression(result['predictions'])
        progress_store[progress_id]['progress'] = 80

        img = Image.open(filepath).convert("RGB")
        draw = ImageDraw.Draw(img)

        damage_classes = [
            "Blue-eye Mold damage", "Drier damage", "Insect damage",
            "Mold damage", "Sprout damage", "Surface Mold", "cracked"
        ]
        heat_damage_classes = ["Heat damage"]
        class_colors = {
            "Normal": "blue",
            "Mold damage": "red",
            "Blue-eye Mold damage": "darkred"
        }

        counts = {}
        for pred in result['predictions']:
            label = pred['class']
            counts[label] = counts.get(label, 0) + 1
            if label not in class_colors:
                class_colors[label] = "#%06x" % random.randint(0, 0xFFFFFF)

            color = class_colors[label]
            x0 = pred['x'] - pred['width'] / 2
            y0 = pred['y'] - pred['height'] / 2
            x1 = pred['x'] + pred['width'] / 2
            y1 = pred['y'] + pred['height'] / 2
            draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
            draw.text((x0, y0 - 10), label, fill=color)

        normal_count = counts.get("Normal", 0)
        total_kernels = sum(counts.values())
        damage_kernels = total_kernels - normal_count

        total_damage_pct = (damage_kernels / total_kernels) * 100 if total_kernels else 0
        heat_damage_count = sum(counts.get(c, 0) for c in heat_damage_classes)
        heat_damage_pct = (heat_damage_count / total_kernels) * 100 if total_kernels else 0

        grade = classify_grade(total_damage_pct, damage_kernels, heat_damage_pct, heat_damage_count, total_kernels)

        output_path = os.path.join(RESULT_FOLDER, filename)
        img.save(output_path)
        progress_store[progress_id]['progress'] = 90

        result_data = {
            'counts': counts,
            'grade': grade,
            'total_damage_pct': round(total_damage_pct, 2),
            'heat_damage_pct': round(heat_damage_pct, 2),
            'total_kernels': total_kernels,
            'filename': filename,
            'raw_result': raw_result
        }
        
        progress_store[progress_id]['result'] = result_data
        progress_store[progress_id]['progress'] = 100
        progress_store[progress_id]['status'] = 'completed'
        
    except Exception as e:
        progress_store[progress_id]['error'] = str(e)
        progress_store[progress_id]['status'] = 'error'

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files['image']
    moisture = float(request.form.get("moisture", 0))
    weight = float(request.form.get("weight", 0))
    
    progress_id = str(uuid.uuid4())
    
    file_ext = os.path.splitext(file.filename)[-1].lower()
    if file_ext in ['.heic', '.heif'] and not HEIC_SUPPORT:
        return jsonify({'error': 'HEIC files not supported. Install pillow-heif.'}), 400
        
    filename = str(uuid.uuid4()) + ('.jpg' if file_ext in ['.heic', '.heif'] else file_ext)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    thread = threading.Thread(target=process_image_async, args=(filepath, filename, moisture, weight, progress_id))
    thread.start()
    
    return jsonify({'progress_id': progress_id})

@app.route("/progress/<progress_id>")
def get_progress(progress_id):
    if progress_id not in progress_store:
        return jsonify({'error': 'Invalid progress ID'}), 404
    
    data = progress_store[progress_id].copy()
    
    if data['status'] == 'completed' and progress_id in progress_store:
        del progress_store[progress_id]
    
    return jsonify(data)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files['image']
        moisture = float(request.form.get("moisture", 0))
        weight = float(request.form.get("weight", 0))

        file_ext = os.path.splitext(file.filename)[-1].lower()
        if file_ext in ['.heic', '.heif'] and not HEIC_SUPPORT:
            return f"<h1>Error: HEIC files not supported</h1><p>Please install pillow-heif or convert to JPG/PNG.</p><a href='/'>Back</a>"

        filename = str(uuid.uuid4()) + ('.jpg' if file_ext in ['.heic', '.heif'] else file_ext)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        compress_image(filepath)

        try:
            raw_result = CLIENT.infer(filepath, model_id=PROJECT_ID)
        except HTTPCallErrorError as e:
            if e.status_code == 413:
                compress_image(filepath, max_size_mb=2)
                try:
                    raw_result = CLIENT.infer(filepath, model_id=PROJECT_ID)
                except HTTPCallErrorError as retry_error:
                    if retry_error.status_code == 413:
                        return f"<h1>Error: Image too large</h1><p>Please upload a smaller image file.</p><a href='/'>Back</a>"
                    raise retry_error
            else:
                raise e
        result = raw_result.copy()
        result['predictions'] = non_max_suppression(result['predictions'])

        img = Image.open(filepath).convert("RGB")
        draw = ImageDraw.Draw(img)

        damage_classes = [
            "Blue-eye Mold damage", "Drier damage", "Insect damage",
            "Mold damage", "Sprout damage", "Surface Mold", "cracked"
        ]
        heat_damage_classes = ["Heat damage"]
        class_colors = {
            "Normal": "blue",
            "Mold damage": "red",
            "Blue-eye Mold damage": "darkred"
        }

        counts = {}
        for pred in result['predictions']:
            label = pred['class']
            counts[label] = counts.get(label, 0) + 1
            if label not in class_colors:
                class_colors[label] = "#%06x" % random.randint(0, 0xFFFFFF)

            color = class_colors[label]
            x0 = pred['x'] - pred['width'] / 2
            y0 = pred['y'] - pred['height'] / 2
            x1 = pred['x'] + pred['width'] / 2
            y1 = pred['y'] + pred['height'] / 2
            draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
            draw.text((x0, y0 - 10), label, fill=color)

        normal_count = counts.get("Normal", 0)
        total_kernels = sum(counts.values())
        damage_kernels = total_kernels - normal_count

        total_damage_pct = (damage_kernels / total_kernels) * 100 if total_kernels else 0
        heat_damage_count = sum(counts.get(c, 0) for c in heat_damage_classes)
        heat_damage_pct = (heat_damage_count / total_kernels) * 100 if total_kernels else 0

        grade = classify_grade(total_damage_pct, damage_kernels, heat_damage_pct, heat_damage_count, total_kernels)

        output_path = os.path.join(RESULT_FOLDER, filename)
        img.save(output_path)

        return render_template_string('''
            <div style="display: flex; flex-direction: row; gap: 40px;">
                <div style="flex: 1;">
                    <h1>Analysis Result</h1>
                    <ul>
                        {% for label, count in counts.items() %}
                            <li>{{ label }}: {{ count }}</li>
                        {% endfor %}
                    </ul>
                    <p><strong>Total Kernels:</strong> {{ total_kernels }}</p>
                    <p><strong>Grade:</strong> {{ grade }}</p>
                    <p><strong>Total damage %:</strong> {{ total_damage_pct | round(2) }}%</p>
                    <p><strong>Heat damage %:</strong> {{ heat_damage_pct | round(2) }}%</p>
                    <h2>Visual</h2>
                    <img src="https://corn-grader-backend.synventra.com/results/{{ filename }}" style="max-width: 100%; height: auto;">
                </div>
                <div style="flex: 1;">
                    <h2><button onclick="this.nextElementSibling.style.display='block'">Show Raw JSON</button></h2>
                    <pre style="display: none">{{ raw_result | tojson(indent=2) }}</pre>
                </div>
            </div>
            <p><a href="/">Back</a></p>
        ''', counts=counts, grade=grade, total_damage_pct=total_damage_pct,
             heat_damage_pct=heat_damage_pct, filename=filename, raw_result=raw_result, total_kernels=total_kernels)

    return '''
        <h1>Upload Corn Kernel Image</h1>
        <form method="post" enctype="multipart/form-data">
            Image: <input type="file" name="image" required><br>
            Weight (lb/bu): <input type="number" step="0.1" name="weight" value="56" required><br>
            Moisture (%): <input type="number" step="0.1" name="moisture" value="15.5" required><br>
            <input type="submit" value="Analyze">
        </form>
    '''

@app.route("/results/<filename>")
def result_image(filename):
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
