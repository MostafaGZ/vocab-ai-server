from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import io
import tensorflow as tf

app = FastAPI(title="Vocab AI Server")

# 1. Load Teachable Machine Model & Labels
model = tf.keras.models.load_model("keras_model.h5", compile=False)
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# 2. Dictionary for translations (Match these exactly to your Teachable Machine labels)
TRANSLATIONS = {
    "Book": {"ar": "كتاب", "en": "book", "es": "libro", "de": "Buch", "fr": "livre"},
    "Cup": {"ar": "كوب", "en": "cup", "es": "taza", "de": "Tasse", "fr": "tasse"},
    "Pen": {"ar": "قلم", "en": "pen", "es": "bolígrafo", "de": "Stift", "fr": "stylo"},
    # Add the rest of your 20 objects here!
}

@app.get("/health")
def health():
    return {"status": "ok"} # Used to keep the server awake

@app.post("/predict")
async def predict(file: UploadFile = File(...), lang: str = Form("en")):
    # Read and preprocess the image
    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB").resize((224, 224))
    
    # Teachable Machine requires normalization to [-1, 1]
    img_array = np.asarray(img, dtype=np.float32)
    normalized_image_array = (img_array / 127.5) - 1
    data = np.expand_dims(normalized_image_array, axis=0)

    # Run AI inference
    prediction = model.predict(data)
    index = np.argmax(prediction)
    confidence = float(prediction[0][index])
    
    # Clean up the label (Teachable Machine adds numbers like "0 Book")
    raw_label = class_names[index]
    clean_label = raw_label.split(" ", 1)[-1].strip() 

    # Get the translation
    word = TRANSLATIONS.get(clean_label, {}).get(lang, clean_label)

    return JSONResponse({
        "success": True,
        "label": clean_label,
        "word": word,
        "confidence": round(confidence, 4),
        "language": lang
    })