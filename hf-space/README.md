---
title: SmartRecycleBin
emoji: ♻️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
models:
  - watersplash/waste-classification
tags:
  - image-classification
  - waste
  - recycling
---

# SmartRecycleBin - Waste Classification

Free Hugging Face Space for SmartRecycleBin. Classifies waste images into 12 categories and maps to bin type.

**Model:** `watersplash/waste-classification` (ViT, 85.8M params, 98% accuracy)  
**Alternatives:** `yangy50/garbage-classification` (6 classes), `kendrickfff/my_resnet50_garbage_classification`

## Env vars (optional, for Supabase)

Set in Space Settings → Variables:

- `SUPABASE_URL` = https://xjjlsxsdjbgrhsnncfdg.supabase.co
- `SUPABASE_KEY` = your anon or service_role key
- `MODEL_ID` = override model (default watersplash/waste-classification)

Space will still work without Supabase — just won't save history.

## Local run

```bash
pip install -r requirements.txt
python app.py
```

## Fine-tune on your data

See `../train.py` to fine-tune ViT on your `category_augmented` folder and push to Hub:

```bash
pip install datasets transformers torch torchvision
python train.py --data_dir "C:/Users/yosf3/OneDrive/Desktop/trash AI/category_augmented" --output your-username/smart-recycle-bin
```
