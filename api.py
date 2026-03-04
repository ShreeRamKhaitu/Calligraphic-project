import os
import io
import numpy as np
import joblib
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional
import data_gen
import train

# Build a single transliteration → Devanagari lookup map
DEVANAGARI_MAP = {**data_gen.CONSONANTS, **data_gen.VOWELS, **data_gen.NUMBERS, **data_gen.SYMBOLS}

app = FastAPI(title="Calligraphic-Python API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Configuration
MODEL_PATH = "ranjana_rf_model.pkl"
CLASSES_PATH = "classes.txt"
DATASET_DIR = "dataset"

# In-memory job state
job_state = {
    "generate": {"status": "idle", "message": ""},
    "train": {"status": "idle", "message": "", "accuracy": None},
}

class StatusResponse(BaseModel):
    model_exists: bool
    dataset_exists: bool
    classes_count: int
    dataset_size: int

def get_dataset_info():
    if not os.path.exists(DATASET_DIR):
        return 0, 0
    classes = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    total_files = sum([len(files) for r, d, files in os.walk(DATASET_DIR)])
    return len(classes), total_files

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/status", response_model=StatusResponse)
async def get_status():
    classes_count, dataset_size = get_dataset_info()
    return StatusResponse(
        model_exists=os.path.exists(MODEL_PATH),
        dataset_exists=os.path.exists(DATASET_DIR),
        classes_count=classes_count,
        dataset_size=dataset_size
    )

@app.get("/job/{job_name}")
async def get_job_status(job_name: str):
    if job_name not in job_state:
        return {"error": "Unknown job"}
    return job_state[job_name]

def run_generate():
    job_state["generate"]["status"] = "running"
    job_state["generate"]["message"] = "Generating dataset..."
    try:
        data_gen.create_dataset()
        job_state["generate"]["status"] = "done"
        job_state["generate"]["message"] = "Dataset generated successfully!"
    except Exception as e:
        job_state["generate"]["status"] = "error"
        job_state["generate"]["message"] = str(e)

def run_train():
    job_state["train"]["status"] = "running"
    job_state["train"]["message"] = "Training model..."
    job_state["train"]["accuracy"] = None
    try:
        accuracy = train.train()
        job_state["train"]["status"] = "done"
        job_state["train"]["accuracy"] = accuracy
        job_state["train"]["message"] = f"Training complete! Accuracy: {accuracy:.1%}"
    except Exception as e:
        job_state["train"]["status"] = "error"
        job_state["train"]["message"] = str(e)

@app.post("/generate")
async def generate_data(background_tasks: BackgroundTasks):
    if job_state["generate"]["status"] == "running":
        return {"message": "Generation already running"}
    job_state["generate"]["status"] = "idle"
    background_tasks.add_task(run_generate)
    return {"message": "Data generation started"}

@app.post("/train")
async def train_model(background_tasks: BackgroundTasks):
    if job_state["train"]["status"] == "running":
        return {"message": "Training already running"}
    job_state["train"]["status"] = "idle"
    background_tasks.add_task(run_train)
    return {"message": "Training started"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASSES_PATH):
        return {"error": "Model not trained yet"}

    # Load model and classes
    model = joblib.load(MODEL_PATH)
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        class_names = f.read().splitlines()

    # Read and process image
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('L')
    img = img.resize((64, 64))
    img_array = np.array(img).flatten() / 255.0
    img_array = img_array.reshape(1, -1)

    # Predict
    prediction = model.predict(img_array)
    probs = model.predict_proba(img_array)
    confidence = float(np.max(probs))
    class_idx = int(prediction[0])
    
    predicted_class = class_names[class_idx]
    devanagari = DEVANAGARI_MAP.get(predicted_class, predicted_class)

    return {
        "predicted_class": predicted_class,
        "devanagari": devanagari,
        "confidence": confidence,
        "all_probs": {class_names[i]: float(probs[0][i]) for i in range(len(class_names))}
    }

class MonogramRequest(BaseModel):
    text: str
    font_size: int = 80
    fg_color: str = "#2d1b69"
    bg_color: Optional[str] = "#ffffff"  # None = transparent
    padding: int = 40
    line_spacing: int = 10
    vertical: bool = True


@app.post("/monogram")
async def generate_monogram(req: MonogramRequest):
    font_path = "NithyaRanjanaDU-Regular.otf"
    text = req.text.strip()
    if not text:
        return {"error": "No text provided"}

    # Transliterate if it's likely Romanized (no Devanagari chars detected)
    import re
    if not re.search(r'[\u0900-\u097F]', text):
        # We need to re-import or re-calculate DEVANAGARI_MAP because it's static
        # For simplicity, move the conversion logic here
        translit_map = {**data_gen.CONSONANTS, **data_gen.VOWELS, **data_gen.NUMBERS, **data_gen.SYMBOLS}
        matra_map = data_gen.MATRAS
        
        # Sort keys by length descending to match longer ones first (kh, aa, etc)
        keys = sorted(translit_map.keys(), key=len, reverse=True)
        vowel_keys = sorted(data_gen.VOWELS.keys(), key=len, reverse=True)
        
        converted = ""
        i = 0
        last_was_consonant = False
        
        while i < len(text):
            match = False
            # Check for vowel signs if the last character was a consonant
            if last_was_consonant:
                for vk in vowel_keys:
                    if text[i:i+len(vk)] == vk:
                        converted += matra_map.get(vk, data_gen.VOWELS[vk])
                        i += len(vk)
                        match = True
                        last_was_consonant = False # Reset context after a matra
                        break
            
            if not match:
                for k in keys:
                    if text[i:i+len(k)] == k:
                        converted += translit_map[k]
                        i += len(k)
                        match = True
                        # Mark if this was a consonant (but not a halant, though halant+vowel is rare)
                        last_was_consonant = (k in data_gen.CONSONANTS and k != '*')
                        break
            
            if not match:
                converted += text[i]
                i += 1
                last_was_consonant = False
        text = converted

    try:
        font = ImageFont.truetype(font_path, req.font_size)
    except Exception as e:
        return {"error": f"Font load failed: {e}"}

    ascent, descent = font.getmetrics()
    # Adding a small buffer to the line height to be safe
    line_height = ascent + descent
    
    # --- Handle Vertical Stacking with Grapheme Clusters ---
    if req.vertical and "\n" not in text:
        import re
        # Regex for Devanagari grapheme clusters:
        # Consonant follow by any number of (Virama + Consonant) and optional (Vowel Sign or other modifiers)
        # Or a standalone Vowel
        cluster_regex = r'[\u0905-\u0939\u0958-\u095F](?:[\u094D][\u0905-\u0939\u0958-\u095F])*[\u093E-\u094C\u094D\u0901-\u0903]?|[\u0905-\u0914]'
        lines = re.findall(cluster_regex, text)
        if not lines: # Fallback if regex fails to match anything for some reason
            lines = list(text)
    else:
        lines = text.split("\n")
    
    # --- Fix for Ranjana Font: Reorder 'i' matra to visual order ---
    # The NithyaRanjana font seems to require 'ि' (\u093F) to be BEFORE the consonant.
    i_matra = '\u093F'
    processed_lines = []
    import re
    cluster_regex = r'[\u0905-\u0939\u0958-\u095F](?:[\u094D][\u0905-\u0939\u0958-\u095F])*[\u093E-\u094C\u094D\u0901-\u0903]?|[\u0905-\u0914]'
    
    for line in lines:
        if i_matra in line:
            # We need to reorder 'ि' to the start of each cluster it belongs to
            # even in horizontal lines.
            clusters = re.findall(cluster_regex, line)
            new_line = ""
            for c in clusters:
                if i_matra in c:
                    new_line += i_matra + c.replace(i_matra, '')
                else:
                    new_line += c
            # Append non-cluster characters (like spaces)
            remaining = line
            for c in clusters:
                remaining = remaining.replace(c, '', 1)
            new_line += remaining # Fallback for spaces/other chars
            processed_lines.append(new_line)
        else:
            processed_lines.append(line)
    lines = processed_lines

    line_bboxes = []
    for line in lines:
        # Use getbbox for most accurate width
        bb = font.getbbox(line) if hasattr(font, 'getbbox') else ddraw.textbbox((0, 0), line, font=font)
        line_bboxes.append(bb)

    line_widths = [bb[2] - bb[0] for bb in line_bboxes]
    total_width = max(line_widths) if line_widths else req.font_size
    
    # Calculate total height based on fixed line height
    total_height = (len(lines) * line_height) + (max(0, len(lines) - 1) * req.line_spacing)

    # Padding and safety buffer for heads
    safety_top = int(req.font_size * 0.15) # 15% extra space at the very top for ascenders
    img_w = total_width + req.padding * 2
    img_h = total_height + req.padding * 2 + safety_top

    # --- Draw ---
    transparent = req.bg_color is None or req.bg_color.lower() == "transparent"
    mode = "RGBA"
    bg = (0, 0, 0, 0) if transparent else (*_hex_to_rgb(req.bg_color), 255)
    img = Image.new(mode, (img_w, img_h), color=bg)
    draw = ImageDraw.Draw(img)

    fg = (*_hex_to_rgb(req.fg_color), 255)
    
    # Start drawing
    y_cursor = req.padding + safety_top
    for i, line in enumerate(lines):
        bb = line_bboxes[i]
        # Center horizontally
        line_w = bb[2] - bb[0]
        x = req.padding + (total_width - line_w) // 2 - bb[0]
        
        # We draw text at y_cursor. 
        # Since PIL default anchor is top, and we've added safety_top, 
        # this should be safe for heads.
        draw.text((x, y_cursor), line, font=font, fill=fg)
        y_cursor += line_height + req.line_spacing

    # --- Return PNG stream ---
    buf = io.BytesIO()
    fmt = "PNG"  # PNG supports transparency
    img.save(buf, format=fmt)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png",
                             headers={"Content-Disposition": 'inline; filename="ranjana_monogram.png"'})


def _hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
