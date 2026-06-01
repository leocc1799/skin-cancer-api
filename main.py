from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import base64
import io
import os
from PIL import Image, ImageDraw
import requests

app = FastAPI(title="Skin Lesion AI API")

# Your updated CORS settings from earlier
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, "allow all" makes it easy to test immediately
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def generate_dummy_heatmap(image: Image.Image) -> str:
    """Keeps the frontend UI happy until you build a real Grad-CAM backend"""
    img = image.convert("RGBA").resize((224, 224))
    overlay = Image.new("RGBA", img.size, (255, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.ellipse((50, 50, 174, 174), fill=(255, 0, 0, 128)) 
    combined = Image.alpha_composite(img, overlay)
    
    buffered = io.BytesIO()
    combined.convert("RGB").save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# --- THE API INTEGRATION ---
# Replace this string with the token you copied from Hugging Face!
HF_TOKEN = os.environ.get("HF_TOKEN") 
HF_API_URL = "https://router.huggingface.co/hf-inference/models/Anwarkh1/Skin_Cancer-Image_Classification"

@app.post("/predict")
async def predict_lesion(file: UploadFile = File(...)):
    try:
        # 1. Read the image sent from Next.js
        contents = await file.read()
        file_type = file.content_type
        # 2. Package it up and send it to the Hugging Face AI model
        headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": file_type  # 👈 Now it adapts to the file uploaded
        }        
        response = requests.post(HF_API_URL, headers=headers, data=contents)
        
        # Catch any API limits or errors
        if response.status_code != 200:
            raise Exception(f"Hugging Face API Error: {response.text}")
            
        ai_results = response.json()
        
        # 3. The API returns a list of guesses. We want the top one (index 0)
        # It looks like this: [{'label': 'Melanoma', 'score': 0.95}, ...]
        top_prediction = ai_results[0]
        
        # 4. Generate the placeholder heatmap for the UI
        image = Image.open(io.BytesIO(contents))
        heatmap_b64 = generate_dummy_heatmap(image)
        
        # 5. Send the REAL AI prediction back to your Next.js app!
        return {
            "class_name": top_prediction["label"],
            "confidence": top_prediction["score"],
            "heatmap_base64": heatmap_b64
        }
        
    except Exception as e:
        print(f"\n❌ SERVER CRASHED! Reason: {str(e)}\n")
        raise HTTPException(status_code=500, detail=f"Backend Error: {str(e)}")
