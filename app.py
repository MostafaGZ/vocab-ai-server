import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import io
import uvicorn
from ai_edge_litert.interpreter import Interpreter

app = FastAPI(title="Vocab AI Server")

# --- Rejection thresholds (tune here, no Pico reflash) ---
MIN_TOP1_CONFIDENCE = 0.75
MIN_TOP1_TOP2_MARGIN = 0.15

# 1. Load model & labels
model_path = "model_unquant.tflite"
interpreter = None
input_details = None
output_details = None
class_names = []

try:
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    with open("labels.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]
    print(f"✅ TFLite loaded. Input {input_details[0]['shape']} / {input_details[0]['dtype']}")
    print(f"✅ {len(class_names)} labels.")
except Exception as e:
    print(f"❌ ERROR LOADING MODEL: {e}")

# 2. Translations — keys match labels.txt; Pen added; typo Hairbrash preserved
TRANSLATIONS = {
    "Ball": {"ar": "الكرة", "en": "ball", "de": "Ball"},
    "Cell phone": {"ar": "هاتف محمول", "en": "cell phone", "de": "Handy"},
    "Shoe": {"ar": "حذاء", "en": "shoe", "de": "Schuh"},
    "Teddy Bear": {"ar": "دبدوب", "en": "teddy bear", "de": "Teddybär"},
    "TV remote": {"ar": "جهاز التحكم", "en": "TV remote", "de": "Fernbedienung"},
    "Bowl": {"ar": "وعاء", "en": "bowl", "de": "Schüssel"},
    "Cup": {"ar": "كوب", "en": "cup", "de": "Tasse"},
    "Plate": {"ar": "طبق", "en": "plate", "de": "Teller"},
    "Spoon": {"ar": "ملعقة", "en": "spoon", "de": "Löffel"},
    "Water bottle": {"ar": "زجاجة ماء", "en": "water bottle", "de": "Wasserflasche"},
    "Hairbrash": {"ar": "فرشاة شعر", "en": "hairbrush", "de": "Haarbürste"},  # matches labels.txt typo
    "Shampoo": {"ar": "شامبو", "en": "shampoo", "de": "Shampoo"},
    "soap": {"ar": "صابون", "en": "soap", "de": "Seife"},
    "Toothbrush": {"ar": "فرشاة أسنان", "en": "toothbrush", "de": "Zahnbürste"},
    "Towel": {"ar": "منشفة", "en": "towel", "de": "Handtuch"},
    "Backpack": {"ar": "حقيبة ظهر", "en": "backpack", "de": "Rucksack"},
    "Book": {"ar": "كتاب", "en": "book", "de": "Buch"},
    "Eraser": {"ar": "ممحاة", "en": "eraser", "de": "Radiergummi"},
    "Notebook": {"ar": "دفتر", "en": "notebook", "de": "Notizbuch"},
    "Pen": {"ar": "قلم", "en": "pen", "de": "Stift"},
    "Ruler": {"ar": "مسطرة", "en": "ruler", "de": "Lineal"},
}
_TRANSLATIONS_CI = {k.strip().lower(): v for k, v in TRANSLATIONS.items()}

def translate(clean_label, lang):
    entry = TRANSLATIONS.get(clean_label) or _TRANSLATIONS_CI.get(clean_label.strip().lower())
    return entry.get(lang, clean_label) if entry else clean_label

def clean(raw_label):
    return raw_label.split(" ", 1)[-1].strip()

def decide(probs):
    order = np.argsort(probs)[::-1]
    i1, i2 = int(order[0]), int(order[1])
    top1, top2 = float(probs[i1]), float(probs[i2])
    margin = top1 - top2
    if top1 < MIN_TOP1_CONFIDENCE:
        return False, "low_confidence", i1, i2, top1, top2, margin
    if margin < MIN_TOP1_TOP2_MARGIN:
        return False, "ambiguous", i1, i2, top1, top2, margin
    return True, "ok", i1, i2, top1, top2, margin

# 3. Endpoints
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": interpreter is not None,
        "labels": len(class_names),
        "min_confidence": MIN_TOP1_CONFIDENCE,
        "min_margin": MIN_TOP1_TOP2_MARGIN,
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...), lang: str = "en"):
    if interpreter is None:
        return JSONResponse({"success": False, "accepted": False,
                             "reason": "no_model", "word": None, "language": lang})
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB").resize((224, 224))
        arr = np.asarray(img, dtype=np.float32)
        arr = (arr / 127.5) - 1.0
        data = np.expand_dims(arr, axis=0).astype(input_details[0]["dtype"])

        interpreter.set_tensor(input_details[0]["index"], data)
        interpreter.invoke()
        probs = interpreter.get_tensor(output_details[0]["index"])[0]

        accepted, reason, i1, i2, top1, top2, margin = decide(probs)
        label = clean(class_names[i1])
        top2_label = clean(class_names[i2])

        if not accepted:
            print(f"🚫 Rejected ({reason}): {label} {top1:.3f} vs {top2_label} {top2:.3f} "
                  f"margin={margin:.3f} lang={lang}")
            return JSONResponse({
                "success": True, "accepted": False, "reason": reason,
                "word": None, "label": label,
                "confidence": round(top1, 4),
                "top2_label": top2_label, "top2_confidence": round(top2, 4),
                "margin": round(margin, 4), "language": lang,
            })

        word = translate(label, lang)
        print(f"📸 Accepted: {label} ({top1:.2%}) -> {word} [{lang}] margin={margin:.3f}")
        return JSONResponse({
            "success": True, "accepted": True, "reason": "ok",
            "label": label, "word": word,
            "confidence": round(top1, 4),
            "top2_label": top2_label, "top2_confidence": round(top2, 4),
            "margin": round(margin, 4), "language": lang,
        })
    except Exception as e:
        print(f"❌ Error: {e}")
        return JSONResponse(status_code=500, content={
            "success": False, "accepted": False, "reason": "bad_image",
            "error": str(e), "language": lang,
        })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
