from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# ---------------------------
# CONFIGURATION
# ---------------------------
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MODEL_PATH = "model/leaf_model.h5"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained model
model = load_model(MODEL_PATH)

# Your class labels from the dataset
CLASS_LABELS = ["Bacterial leaf blight", "Brown spot", "Leaf smut"]

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ---------------------------
# ROUTES
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return render_template("index.html", error="No file uploaded.")

        file = request.files["image"]

        if file.filename == "":
            return render_template("index.html", error="No file selected.")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Preprocess image
            img_array = preprocess_image(filepath)

            # Predict
            prediction = model.predict(img_array)[0]
            confidence = float(np.max(prediction))
            species = CLASS_LABELS[np.argmax(prediction)]

            return render_template(
                "index.html",
                species=species,
                confidence=round(confidence, 2),
                uploaded_image=filename
            )
        else:
            return render_template("index.html", error="Invalid file type.")
    return render_template("index.html")

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)
