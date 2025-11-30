# train_text.py
from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np

MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 16
EPOCHS = 3
NUM_LABELS = 3  # update according to your dataset

def preprocess_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, max_length=256)

def main():
    # Load labeled CSV files
    ds = load_dataset("csv", data_files={
        "train": "data/text_train.csv",
        "validation": "data/text_val.csv"
    })
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = ds.map(lambda e: preprocess_function(e, tokenizer), batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=NUM_LABELS
    )

    # Load metric using evaluate
    metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return metric.compute(predictions=preds, references=labels, average="weighted")

    training_args = TrainingArguments(
        output_dir="outputs/text_model",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        learning_rate=2e-5,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=False,  # macOS CPU doesn't support fp16
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model("outputs/text_model/best")
    tokenizer.save_pretrained("outputs/text_model/best")

if __name__ == "__main__":
    main()
