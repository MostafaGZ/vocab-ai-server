from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import io
import tensorflow as tf
import os
import uvicorn

app = FastAPI(title="Vocab AI Server")

# 1. Load Teachable Machine Model & Labels
try:
    model = tf.keras.models.load_model("keras_model.h5", compile=False)
    with open("labels.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    print("✅ Model and labels loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# 2. Dictionary for translations (English, Arabic, German)
# IMPORTANT: These keys must perfectly match the names you used in Teachable Machine!
TRANSLATIONS = {
    "Ball": {"ar": "الكرة", "en": "ball", "de": "Ball"},
    "Cell Phone": {"ar": "هاتف محمول", "en": "cell phone", "de": "Handy"},
    "Shoe": {"ar": "حذاء", "en": "shoe", "de": "Schuh"},
    "Teddy Bear": {"ar": "دبدوب", "en": "teddy bear", "de": "Teddybär"},
    "TV Remote": {"ar": "جهاز التحكم", "en": "TV remote", "de": "Fernbedienung"},
    "Bowl": {"ar": "وعاء", "en": "bowl", "de": "Schüssel"},
    "Cup": {"ar": "كوب", "en": "cup", "de": "Tasse"},
    "Plate": {"ar": "طبق", "en": "plate", "de": "Teller"},
    "Spoon": {"ar": "ملعقة", "en": "spoon", "de": "Löffel"},
    "Water Bottle": {"ar": "زجاجة ماء", "en": "water bottle", "de": "Wasserflasche"},
    "Hairbrush": {"ar": "فرشاة شعر", "en": "hairbrush", "de": "Haarbürste"},
    "Shampoo": {"ar": "شامبو", "en": "shampoo", "de": "Shampoo"},
    "Soap": {"ar": "صابون", "en": "soap", "de": "Seife"},
    "Toothbrush": {"ar": "فرشاة أسنان", "en": "toothbrush", "de": "Zahnbürste"},
    "Towel": {"ar": "منشفة", "en": "towel", "de": "Handtuch"},
    "Backpack": {"ar": "حقيبة ظهر", "en": "backpack", "de": "Rucksack"},
    "Book": {"ar": "كتاب", "en": "book", "de": "Buch"},
    "Eraser": {"ar": "ممحاة", "en": "eraser", "de": "Radiergummi"},
    "Notebook": {"ar": "دفتر", "en": "notebook", "de": "Notizbuch"},
    "Ruler": {"ar": "مسطرة", "en": "ruler", "de": "Lineal"}
}

@app.get("/health")
def health():
    # Used by UptimeRobot to keep the server awake
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), lang: str = "en"):
    try:
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

        # Get the translation based on the 'lang' parameter (defaults to English)
        word = TRANSLATIONS.get(clean_label, {}).get(lang, clean_label)

        # Print to logs so you can see it working in the Netrun dashboard!
        print(f"📸 Image received. AI guessed: {clean_label} ({confidence:.2%}). Language requested: {lang} -> {word}")

        return JSONResponse({
            "success": True,
            "label": clean_label,
            "word": word,
            "confidence": round(confidence, 4),
            "language": lang
        })
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

# This block allows the server to run on hosting platforms
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
