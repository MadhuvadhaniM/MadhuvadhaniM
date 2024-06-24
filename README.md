from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import torch

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Example dataset (replace with your own)
train_dataset = [
    "Text 1 to fine-tune GPT-2.",
    "Another example sentence.",
    "More text for training."
]

# Tokenize the dataset
train_tokenized = tokenizer(train_dataset, return_tensors="pt", truncation=True, padding=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=1000,
    save_total_limit=2,
    prediction_loss_only=True,
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized
)

# Start training
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine-tuned-gpt2")

