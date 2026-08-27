"""
Fine-tune ViT on your local garbage dataset and push to Hugging Face Hub.
Dataset structure expected (same as ML.NET folder):
  category_augmented/
    cardboard/*.jpg
    glass/*.jpg
    metal/*.jpg
    ...

Usage:
  pip install datasets transformers torch torchvision evaluate
  python train.py --data_dir "C:/Users/yosf3/OneDrive/Desktop/trash AI/category_augmented" --output your-username/smart-recycle-bin
  # Or skip --output to just save locally to ./finetuned
"""
import argparse
import os
from datasets import load_dataset
from transformers import (
    AutoImageProcessor, AutoModelForImageClassification,
    TrainingArguments, Trainer
)
import evaluate
import torch

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="Path to category_augmented folder")
    p.add_argument("--base_model", default="google/vit-base-patch16-224", help="Base ViT")
    p.add_argument("--output", default="./finetuned", help="Hub repo id or local path")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    return p.parse_args()

def main():
    args = parse_args()
    print(f"Data dir: {args.data_dir}")
    print(f"Base: {args.base_model} -> Output: {args.output}")

    # Load imagefolder dataset
    ds = load_dataset("imagefolder", data_dir=args.data_dir)
    # ds has 'train' split only - create 80/20 split
    ds = ds["train"].train_test_split(test_size=0.2, seed=42)
    labels = ds["train"].features["label"].names
    print(f"Labels ({len(labels)}): {labels}")
    print(f"Train: {len(ds['train'])}, Test: {len(ds['test'])}")

    id2label = {i: l for i, l in enumerate(labels)}
    label2id = {l: i for i, l in id2label.items()}

    processor = AutoImageProcessor.from_pretrained(args.base_model)

    def transforms(examples):
        # examples["image"] is PIL
        inputs = processor([img.convert("RGB") for img in examples["image"]], return_tensors="pt")
        # processor returns pixel_values as tensor - need list per example
        examples["pixel_values"] = [v for v in inputs["pixel_values"]]
        return examples

    ds = ds.with_transform(transforms)

    def collate_fn(examples):
        pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
        labels_t = torch.tensor([ex["label"] for ex in examples])
        return {"pixel_values": pixel_values, "labels": labels_t}

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        preds = eval_pred.predictions.argmax(axis=1)
        return accuracy.compute(predictions=preds, references=eval_pred.label_ids)

    model = AutoModelForImageClassification.from_pretrained(
        args.base_model,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=args.output if not "/" in args.output else "./tmp_train",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,  # set True if you want auto push
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=processor,
    )

    trainer.train()
    print("Training done. Evaluating...")
    print(trainer.evaluate())

    # Save
    save_path = args.output
    if "/" in args.output:  # looks like hub id
        # Save locally then push
        local = "./finetuned"
        trainer.save_model(local)
        processor.save_pretrained(local)
        print(f"Saved locally to {local}")
        print(f"To push to Hub: huggingface-cli login && trainer.push_to_hub('{args.output}')")
        # Auto push if logged in
        try:
            trainer.push_to_hub(args.output)
            processor.push_to_hub(args.output)
            print(f"Pushed to https://huggingface.co/{args.output}")
        except Exception as e:
            print(f"Push failed (login required): {e}")
            print(f"Run: huggingface-cli login")
    else:
        trainer.save_model(save_path)
        processor.save_pretrained(save_path)
        print(f"Saved to {save_path}")

if __name__ == "__main__":
    main()
