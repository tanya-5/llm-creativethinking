import json
import json_lines
import numpy as np
import torch

from datasets import Dataset
from collections import defaultdict
from sklearn.metrics import accuracy_score

from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq


# Function to generate a prompt for the riddle sense task
def get_riddlesense_prompt(question, options):
    prompt = \
        """
Question: {}

What is the correct answer to the question from the following choices?
Options: 
(A): {}
(B): {}
(C): {}
(D): {}
(E): {}""".format(question, options[0], options[1], options[2], options[3], options[4])
    return prompt

# Function to load original data from a JSON lines file
def load_og_data(file_path):
    raw_data = []
    with open(file_path, 'rb') as f:
        for item in json_lines.reader(f):
            raw_data.append(item)

    data = defaultdict(list)
    for item in raw_data:
        data['question'].append(item['question']['stem'])
        data['options'].append([_['text']
                               for _ in item['question']['choices']])
        data['answer'].append(item['answerKey'])
    return data

# Mapping numerical labels to answer options
answer_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}


# Function to load adversarial data from a JSON file (Semantic and Context Reconstructions)
def load_adversarial_data(file_path):
    with open(file_path, 'r') as f:
        raw_data = json.load(f)

    data = defaultdict(list)
    for item in raw_data:
        data['question'].append(item['question'])
        # data['options'].append(item['choice_list'])
        data['options'].append([str(option) for option in item['choice_list']])
        data['answer'].append(answer_map[item['label']])
    return data


# Function to preprocess data samples for the model
def preprocess_function(sample):
    text = get_riddlesense_prompt(sample['question'], sample['options'])

    model_inputs = tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding='max_length',
        return_attention_mask=True,
    )

    labels = tokenizer(sample['answer'], max_length=2,
                       padding="max_length", truncation=True)
    labels = labels["input_ids"]
    labels = [l if l != tokenizer.pad_token_id else -100 for l in labels]
    model_inputs["labels"] = labels

    return model_inputs

# Load the model and PEFT configuration
print("loading model and peft config...")
model_name = 'google/flan-t5-xl'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define LoRA Config
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

# Add LoRA adapter to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


print("loading and processing data...")
# train_data = load_og_data("data/rs_train.jsonl")
# valid_data = load_og_data("data/rs_dev.jsonl")

train_data = load_adversarial_data("data/adversarial_rs_train.json")
valid_data = load_og_data("data/rs_dev.jsonl")

train_dataset = Dataset.from_dict(train_data)
train_tokenized = train_dataset.map(preprocess_function, batched=False, remove_columns=[
                                    'question', 'options', 'answer'])
print(train_tokenized)

valid_dataset = Dataset.from_dict(valid_data)
valid_tokenized = valid_dataset.map(preprocess_function, batched=False, remove_columns=[
                                    'question', 'options', 'answer'])
print(valid_tokenized)

# Define label pad token ID to ignore pad token in the loss
label_pad_token_id = -100

# Data collator for sequence-to-sequence tasks
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=2
)


# Function to compute metrics for evaluation
def compute_metrics(p):
    predictions, labels = p
    # https://discuss.huggingface.co/t/what-does-evalprediction-predictions-contain-exactly/1691/4
    logits = predictions[0]
    predictions = np.argmax(logits, axis=2)

    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100 and l != 1]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [l for (p, l) in zip(prediction, label) if l != -100 and l != 1]
        for prediction, label in zip(predictions, labels)
    ]

    results = accuracy_score(y_true=true_labels, y_pred=true_predictions)
    return {
        "accuracy": results,
    }


print("defining training args and trainer...")
output_dir = f"/usr1/data/devanshj/brainteaser/checkpoints/{model_name[7:]}_adv_lora"

# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    learning_rate=3e-4,  # higher learning rate
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=500,
    load_best_model_at_end=True,
    push_to_hub=True
)


# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=valid_tokenized,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

print("training...")
trainer.train()

print("pushing to hub...")
trainer.push_to_hub()

# Save the trained model and tokenizer
print("saving model...")
peft_model_id = f"/usr1/data/devanshj/brainteaser/checkpoints/{model_name[7:]}_adv_lora_adapter"
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)
