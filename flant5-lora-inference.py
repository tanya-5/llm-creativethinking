import json
import pandas as pd
import torch
import tqdm

from collections import defaultdict
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Mapping answer options to numeric labels
option2label = {
    '(A)': 0,
    'A': 0,
    '(B)': 1,
    'B': 1,
    '(C)': 2,
    'C': 2,
    '(D)': 3,
    'D': 3
}


# Load training data from JSON file
file_path = "data/SP-train.json"
with open(file_path) as f:
    data = json.load(f)
print("number of puzzles: ", len(data))

# # Load eval data from JSON file
# file_path = "SP-eval.json"
# with open(file_path) as f:
#     data = json.load(f)
# print(len(data))

# Check the available device (GPU or CPU) for model computations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# Define paths for different models
model_paths = {
    'flan-t5-xl_rs_lora': '/usr1/data/devanshj/brainteaser/checkpoints/flan-t5-xl_rs_lora_adapter',
    'flan-t5-xl_adv_lora': '/usr1/data/devanshj/brainteaser/checkpoints/flan-t5-xl_adv_lora_adapter',
}

# Iterate over each model
for model_name, model_path in model_paths.items():
    print("-" * 50)
    print("model name:", model_name)

    print("loading peft model...")
    peft_model_id = model_path
    config = PeftConfig.from_pretrained(peft_model_id)
    # MODEL_SIZE = 'xl'
    # BASE_MODEL = f"google/flan-t5-{MODEL_SIZE}"

    # Load the model and move it to the device
    # Load the corresponding tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, peft_model_id).to(device)
    model.eval()
    print("peft model loaded...")

    prediction_data = defaultdict(list)

    # Iterate over each multiple-choice question (MCQ) in the data
    for mcq in tqdm.tqdm(data):
        prediction_data['id'].append(mcq['id'])
        prediction_data['question'].append(mcq['question'])
        prediction_data['answer'].append(mcq['answer'])
        prediction_data['label'].append(mcq['label'])
        prediction_data['choice_0'].append(mcq['choice_list'][0])
        prediction_data['choice_1'].append(mcq['choice_list'][1])
        prediction_data['choice_2'].append(mcq['choice_list'][2])
        prediction_data['choice_3'].append(mcq['choice_list'][3])
        if 'SR' in mcq['id']:
            prediction_data['group'].append('SR')
        elif 'CR' in mcq['id']:
            prediction_data['group'].append('CR')
        else:
            prediction_data['group'].append('OG')

        # Create the prompt for the model
        prompt = f"""Question: {mcq['question']}

        What is the correct answer to the question from the following choices?
        (A) {mcq['choice_list'][0]}
        (B) {mcq['choice_list'][1]}
        (C) {mcq['choice_list'][2]}
        (D) {mcq['choice_list'][3]}"""

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate the model's prediction and decode predicted output
        outputs = model.generate(**inputs)
        predicted_option = tokenizer.batch_decode(
            outputs, skip_special_tokens=True)[0]

        # Map the predicted option to the corresponding label
        for k, v in option2label.items():
            if k in predicted_option:
                predicted_label = v

        prediction_data['prediction'].append(predicted_label)
        # Check if the prediction is correct
        if int(mcq['label']) == int(predicted_label):
            prediction_data['correct'].append(1)
        else:
            prediction_data['correct'].append(0)
    prediction_df = pd.DataFrame(prediction_data)

    # Calculate overall accuracy
    accuracy = prediction_df['correct'].sum() / prediction_df.shape[0]
    print("overall accuracy: ", accuracy * 100)

    # Save the prediction results to a CSV file
    print("storing results...")
    prediction_df.to_csv(f"results/finetuned/{model_name}.csv")
