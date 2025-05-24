import math
import pandas as pd
import torch
import argparse # Added argparse
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    AdamW,
    get_linear_schedule_with_warmup
)
from datasets import Dataset as HFDataset
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# ---------------------------
# Parameters (Easy to Adjust)
# ---------------------------
# MODEL_NAME is still hardcoded as it's a base model, not a target for this change.
MODEL_NAME = "bert-base-uncased"  # Base model to fall back to if no checkpoint exists

# ---------------------------
# 3. Custom Logging Callback for Tracking (defined globally as it's a class)
# ---------------------------
class CustomLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            print(f"Step {state.global_step}: {logs}")

# ---------------------------
# 6. Compute Metrics Function (defined globally)
# ---------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    pred_labels = np.argmax(logits, axis=1)
    f1 = f1_score(labels, pred_labels)
    precision = precision_score(labels, pred_labels)
    recall = recall_score(labels, pred_labels)
    accuracy = accuracy_score(labels, pred_labels)
    return {
        "eval_accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

def main(hub_model_id_arg, start_index_arg):
    # ---------------------------
    # 1. Load and Prepare Data
    # ---------------------------
    df = pd.read_parquet("hf://datasets/latterworks/nuforc-parquet/parquetNUFORCDB1.parquet")
    df_clf = df[["Narrative", "HighlightedReport"]].dropna(subset=["Narrative", "HighlightedReport"]).copy()
    df_clf["Narrative"] = df_clf["Narrative"].astype(str).str.strip()
    df_clf = df_clf[(df_clf["Narrative"] != "") & (df_clf["HighlightedReport"].astype(str).str.strip() != "")]
    df_clf["HighlightedReport"] = df_clf["HighlightedReport"].astype(int)

    # Select data from the start_index_arg onwards
    df_train = df_clf.iloc[start_index_arg:].copy()
    print(f"Number of training samples: {len(df_train)}")

    hf_dataset = HFDataset.from_pandas(df_train)

    # ---------------------------
    # 2. Preprocessing Function for Classification
    # ---------------------------
    max_length = 512
    # MODEL_NAME is used here (global)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    def preprocess_classification(examples):
        encodings = tokenizer(
            examples["Narrative"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )
        encodings["labels"] = examples["HighlightedReport"]
        # Retain original Narrative for evaluation.
        encodings["Narrative"] = examples["Narrative"]
        return encodings

    tokenized_dataset = hf_dataset.map(
        preprocess_classification,
        batched=True,
        remove_columns=["HighlightedReport"]
    )

    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    # ---------------------------
    # 4. Load Model & Setup Trainer
    # ---------------------------
    # Attempt to load the previously fine-tuned model from the Hub.
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            hub_model_id_arg, # Use argument here
            num_labels=2,
            trust_remote_code=True
        )
        print("✅ Loaded fine-tuned model from Hub:", hub_model_id_arg)
    except Exception as e:
        print(f"⚠️ Could not load model from Hub '{hub_model_id_arg}', falling back to base model '{MODEL_NAME}'. Error:", e)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, # Use global MODEL_NAME as fallback
            num_labels=2,
            trust_remote_code=True
        )

    if torch.cuda.is_available():
        model.to("cuda")
        print("✅ CUDA available: model moved to GPU.")

    # ---------------------------
    # 5. Training Arguments with Improvements
    # ---------------------------
    training_args = TrainingArguments(
        output_dir=f"./{hub_model_id_arg.split('/')[-1]}", # Dynamic output dir based on hub_model_id
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=1000,
        eval_steps=100,
        logging_steps=100,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        learning_rate=1e-5,
        num_train_epochs=10,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=True,
        push_to_hub=True,
        hub_model_id=hub_model_id_arg, # Use argument here
        load_best_model_at_end=True,
        metric_for_best_model='eval_accuracy',
        greater_is_better=True,
        report_to=["tensorboard"],
    )

    # ---------------------------
    # 7. Setup Optimizer and Scheduler
    # ---------------------------
    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)
    num_training_steps = (len(train_dataset) * training_args.num_train_epochs) // training_args.per_device_train_batch_size
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # ---------------------------
    # 8. Initialize Trainer
    # ---------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=[CustomLoggingCallback()], # Global class
        compute_metrics=compute_metrics,    # Global function
        optimizers=(optimizer, scheduler)
    )

    # ---------------------------
    # 9. Fine-Tune, Resume if Applicable, and Push Checkpoints to Hub
    # ---------------------------
    resume_checkpoint = None
    print("🚀 Starting fine-tuning (continuing from previous training if a checkpoint is provided)...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # Save model and tokenizer locally
    model.save_pretrained(f"./{hub_model_id_arg.split('/')[-1]}") # Dynamic output dir
    tokenizer.save_pretrained(f"./{hub_model_id_arg.split('/')[-1]}") # Dynamic output dir

    # ---------------------------
    # 10. Evaluate, Compute Perplexity, and Push to Hub
    # ---------------------------
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    trainer.push_to_hub()
    print("✅ Fine-tuning complete. Model saved locally and pushed to the Hub.")

    # ---------------------------
    # 11. Evaluate and Display Predictions with Additional Metrics
    # ---------------------------
    predictions_output = trainer.predict(eval_dataset)
    pred_logits = predictions_output.predictions
    pred_labels = np.argmax(pred_logits, axis=1)
    true_labels = np.array(eval_dataset["labels"])

    correct_predictions = np.sum(pred_labels == true_labels)
    accuracy_val = correct_predictions / len(true_labels)
    f1 = f1_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)

    print("\n📊 Validation Score:")
    print(f"  Correct Predictions : {correct_predictions} out of {len(true_labels)}")
    print(f"  Accuracy            : {accuracy_val:.2%}")
    print(f"  F1-score            : {f1:.2%}")
    print(f"  Precision           : {precision:.2%}")
    print(f"  Recall              : {recall:.2%}")

    print("\n--- Sample Predictions on Validation Data ---")
    for idx in range(min(50, len(eval_dataset))):
        narrative = eval_dataset["Narrative"][idx] # Narrative is already part of eval_dataset due to preprocess_classification
        print(f"Example {idx+1}:")
        print(f"  Narrative      : {narrative}")
        print(f"  Ground Truth   : {true_labels[idx]}")
        print(f"  Prediction     : {pred_labels[idx]}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model for classifying NUFORC highlighted reports.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default="latterworks/highlightedreport-classifier-test",
        help="Hugging Face Hub model ID to push to (e.g., your-username/your-model-name)."
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=100000,
        help="Starting index for slicing the training data from the NUFORC dataset."
    )
    args = parser.parse_args()
    main(args.hub_model_id, args.start_index)
