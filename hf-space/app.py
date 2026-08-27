import os
import gradio as gr
from transformers import pipeline
from PIL import Image
import json
from datetime import datetime

# Supabase optional - only if env vars set
supabase = None
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
if supabase_url and supabase_key:
    try:
        from supabase import create_client
        supabase = create_client(supabase_url, supabase_key)
        print(f"Supabase connected: {supabase_url}")
    except Exception as e:
        print(f"Supabase init failed: {e}")

# Load model - 12 classes, 98% accuracy, 85.8M params
# Options:
# - watersplash/waste-classification (12 classes, ViT, 98% acc) - default
# - yangy50/garbage-classification (6 classes, ViT)
# - kendrickfff/my_resnet50_garbage_classification (12 classes, ResNet50)
MODEL_ID = os.getenv("MODEL_ID", "watersplash/waste-classification")

print(f"Loading model: {MODEL_ID} ...")
classifier = pipeline("image-classification", model=MODEL_ID)
print("Model loaded!")

# Label to bin mapping for SmartRecycleBin
BIN_MAP = {
    "Battery": "hazardous",
    "Biological": "organic",
    "Brown-glass": "glass",
    "Cardboard": "paper",
    "Clothes": "textile",
    "Green-Glass": "glass",
    "Metal": "metal",
    "Paper": "paper",
    "Plastic": "plastic",
    "Shoes": "textile",
    "Trash": "general",
    "White-Glass": "glass",
    # yangy50 6-class fallback
    "cardboard": "paper",
    "glass": "glass",
    "metal": "metal",
    "paper": "paper",
    "plastic": "plastic",
    "trash": "general",
}

def classify(image, save_to_db=False):
    if image is None:
        return "Please upload an image.", {}, ""

    # Gradio gives PIL or numpy - ensure PIL
    if isinstance(image, str):
        image = Image.open(image)
    elif not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    results = classifier(image)
    # results: [{"label": "Plastic", "score": 0.98}, ...]

    top = results[0]
    predicted = top["label"]
    confidence = top["score"]
    bin_type = BIN_MAP.get(predicted, "unknown")

    # All scores dict
    all_scores = {r["label"]: round(r["score"], 4) for r in results}
    scores_json = json.dumps(all_scores, indent=2)

    # Save to Supabase if requested and connected
    db_status = ""
    if save_to_db and supabase:
        try:
            # Note: requires auth - for Space we save anonymously or with service_role
            # If SUPABASE_SERVICE_KEY is set, we can insert directly
            data = {
                "image_path": "hf-space-upload",
                "predicted_label": predicted,
                "confidence_score": float(confidence),
                "all_scores": all_scores,
            }
            # Add user_id if available from auth (Space has no auth)
            result = supabase.table("predictions").insert(data).execute()
            db_status = "Saved to Supabase!"
        except Exception as e:
            db_status = f"Supabase save failed: {e}"
    elif save_to_db and not supabase:
        db_status = "Supabase not configured (set SUPABASE_URL + SUPABASE_KEY)."

    summary = f"**Predicted:** {predicted} ({confidence:.2%}) → Bin: **{bin_type}**"
    if db_status:
        summary += f"\n\n{db_status}"

    return summary, all_scores, scores_json


# Gradio UI
with gr.Blocks(title="SmartRecycleBin - Waste Classification") as demo:
    gr.Markdown("""
    # SmartRecycleBin - Hugging Face Space
    Upload a waste image to classify it. Model: `watersplash/waste-classification` (ViT, 12 classes, 98% accuracy).
    Free hosting on Hugging Face — no card required. Connected to Supabase for history.
    """)

    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Upload waste image")
            save_chk = gr.Checkbox(label="Save prediction to Supabase", value=False)
            btn = gr.Button("Classify", variant="primary")
        with gr.Column():
            out_text = gr.Markdown(label="Result")
            out_label = gr.Label(label="All scores")
            out_json = gr.Code(label="Scores JSON", language="json")

    btn.click(fn=classify, inputs=[inp, save_chk], outputs=[out_text, out_label, out_json])
    inp.change(fn=classify, inputs=[inp, save_chk], outputs=[out_text, out_label, out_json])

    gr.Markdown("""
    **Classes:** Battery, Biological, Brown-glass, Cardboard, Clothes, Green-Glass, Metal, Paper, Plastic, Shoes, Trash, White-Glass
    **API:** POST `/api/predict` with `{"data": [image]}` or use Gradio client: `from gradio_client import Client; Client("YOUR_SPACE_URL").predict(image)`
    """)

if __name__ == "__main__":
    demo.launch()
