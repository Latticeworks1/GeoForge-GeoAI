import math
import pandas as pd
import torch
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
# TRAIN_SAMPLE_SIZE = 1000  # You might not need this anymore
HUB_MODEL_ID = "latterworks/highlightedreport-classifier-test"  # Your HF Hub repository ID
MODEL_NAME = "bert-base-uncased"  # Base model to fall back to if no checkpoint exists
START_INDEX = 100000 # Define the starting index

# ---------------------------
# 1. Load and Prepare Data
# ---------------------------
df = pd.read_parquet("hf://datasets/latterworks/nuforc-parquet/parquetNUFORCDB1.parquet")
df_clf = df[["Narrative", "HighlightedReport"]].dropna(subset=["Narrative", "HighlightedReport"]).copy()
df_clf["Narrative"] = df_clf["Narrative"].astype(str).str.strip()
df_clf = df_clf[(df_clf["Narrative"] != "") & (df_clf["HighlightedReport"].astype(str).str.strip() != "")]
df_clf["HighlightedReport"] = df_clf["HighlightedReport"].astype(int)

# Select data from the START_INDEX onwards
df_train = df_clf.iloc[START_INDEX:].copy()
print(f"Number of training samples: {len(df_train)}")

# If you still want to limit the size, you can add a head() here:
# df_small = df_train.head(TRAIN_SAMPLE_SIZE).copy()
# hf_dataset = HFDataset.from_pandas(df_small)
hf_dataset = HFDataset.from_pandas(df_train)

# ---------------------------
# 2. Preprocessing Function for Classification
# ---------------------------
max_length = 512
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
# 3. Custom Logging Callback for Tracking
# ---------------------------
class CustomLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            print(f"Step {state.global_step}: {logs}")

# ---------------------------
# 4. Load Model & Setup Trainer
# ---------------------------
# Attempt to load the previously fine-tuned model from the Hub.
try:
    model = AutoModelForSequenceClassification.from_pretrained(
        HUB_MODEL_ID,
        num_labels=2,
        trust_remote_code=True
    )
    print("✅ Loaded fine-tuned model from Hub.")
except Exception as e:
    print("⚠️ Could not load model from Hub, falling back to base model. Error:", e)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
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
    output_dir="./highlightedreport-classifier-test",
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=1000,      # Save frequently on a small dataset.
    eval_steps=100,
    logging_steps=100,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    learning_rate=1e-5,
    num_train_epochs=10,
    warmup_ratio=0.1,   # Warmup for 10% of training steps
    weight_decay=0.01,
    fp16=True,  # Enable mixed precision training
    push_to_hub=True,
    hub_model_id=HUB_MODEL_ID,
    load_best_model_at_end=True,
    metric_for_best_model='eval_accuracy',
    greater_is_better=True,
    report_to=["tensorboard"],
)

# ---------------------------
# 6. Compute Metrics Function
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
    callbacks=[CustomLoggingCallback()],
    compute_metrics=compute_metrics,
    optimizers=(optimizer, scheduler)
)

# ---------------------------
# 9. Fine-Tune, Resume if Applicable, and Push Checkpoints to Hub
# ---------------------------
# If you want to resume from a specific local checkpoint, set resume_checkpoint accordingly.
resume_checkpoint = None  # e.g., "./highlightedreport-classifier-test/checkpoint-50"
print("🚀 Starting fine-tuning (continuing from previous training if a checkpoint is provided)...")
trainer.train(resume_from_checkpoint=resume_checkpoint)

# Save model and tokenizer locally
model.save_pretrained("./highlightedreport-classifier-test")
tokenizer.save_pretrained("./highlightedreport-classifier-test")

# ---------------------------
# 10. Evaluate, Compute Perplexity, and Push to Hub
# ---------------------------
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

# Push the model to the Hugging Face Hub.
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
    narrative = eval_dataset["Narrative"][idx]
    print(f"Example {idx+1}:")
    print(f"  Narrative      : {narrative}")
    print(f"  Ground Truth   : {true_labels[idx]}")
    print(f"  Prediction     : {pred_labels[idx]}\n")
