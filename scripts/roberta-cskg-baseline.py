import os
import json
import tqdm
import torch
import logging
import argparse
import numpy as np
from torch.nn import CrossEntropyLoss
from transformers import RobertaTokenizer, RobertaForMaskedLM

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
MAX_SEQUENCE_PER_TIME = 80

class BrainteaserInstanceReader(object):
    """
    Reads the CommonsenseQA dataset into a unified format with context, question, label, and choices.
    """
    def to_uniform_fields(self, fields):
        context = ''
        question = 'Q: '+ fields['question']
        # label = ['A','B','C','D'].index(fields['choice']) if "answerKey" in fields else None
        label = fields['label']
        choices = ['A: '+ c.lower() for c in fields['choice_list']]
        return context, question, label, choices
    
    def fields_to_instance(self, fields):
        context, question, label, choices = self.to_uniform_fields(fields)
        return context, question, label, choices


def token_wise_scoring(sequences, label_ids, attention_mask, tokenizer, device, model):
    choice_loss = [0 for i in range(len(sequences))]
    for i in range(len(sequences)):
        tmp_seq_list = []
        tmp_label_list = []
        tmp_attention_mask = []
        curr_label_ids = label_ids[i]
        for j, t in enumerate(curr_label_ids):
            if t == -100:
                continue
            tmp_seq = torch.tensor(sequences[i][:j]+[tokenizer.mask_token_id]+sequences[i][j+1:]).long().to(device)
            tmp_label = torch.tensor([-100]*j+sequences[i][j:j+1]+[-100]*(len(sequences[i])-j-1)).long().to(device)
            tmp_seq_list.append(tmp_seq)
            tmp_label_list.append(tmp_label)
            tmp_attention_mask.append(torch.tensor(attention_mask[i]).long().to(device))
        tmp_seq_list = torch.stack(tmp_seq_list)
        tmp_label_list = torch.stack(tmp_label_list)
        tmp_attention_mask = torch.stack(tmp_attention_mask)
        if len(tmp_seq_list) < MAX_SEQUENCE_PER_TIME:
            loss = get_lm_score(model, tmp_seq_list, tmp_label_list, tmp_attention_mask)
        else:
            loss = []
            for chunk in range(0, len(tmp_seq_list), MAX_SEQUENCE_PER_TIME):
                loss.append(get_lm_score(model, tmp_seq_list[chunk:chunk+MAX_SEQUENCE_PER_TIME], tmp_label_list[chunk:chunk+MAX_SEQUENCE_PER_TIME], tmp_attention_mask[chunk:chunk+MAX_SEQUENCE_PER_TIME]))
            loss = np.concatenate(loss)
        choice_loss[i] = sum(loss)/len(loss) 
    prediction = choice_loss.index(min(choice_loss))
    return prediction

def prepare_input(sequences, label_ids, pad_token_id):
    max_length = max([len(text) for text in sequences])
    attention_mask = np.zeros((len(sequences), max_length))
    for i in range(len(sequences)):
        attention_mask[i][:len(sequences[i])] = 1
    sequences = [text + [pad_token_id] * (max_length - len(text)) for text in sequences]
    label_ids = [text + [-100] * (max_length - len(text)) for text in label_ids]
    return sequences, label_ids, attention_mask

def score_task(question, choices, tokenizer, device, model):
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    question_ids = tokenizer.encode(question)
    choice_ids = [tokenizer.encode(choice, add_prefix_space=True)[1:-1] for choice in choices]
    sequences = [question_ids[:-1] + choice_ids[i] +[tokenizer.sep_token_id] for i in range(len(choice_ids))]
    label_ids = [[-100]+text[1:-1]+[-100] for text in sequences]
    sequences, label_ids, attention_mask = prepare_input(sequences, label_ids, pad_token_id)
    prediction = token_wise_scoring(sequences, label_ids, attention_mask, tokenizer, device, model)
    return prediction

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm", default="roberta-large", type=str, required=False, help="language model to use")
    parser.add_argument("--dataset_file", default=None, type=str, required=True, help="Jsonl file")
    parser.add_argument("--out_dir", default=None, type=str, required=True, help="Out directory for the predictions")
    parser.add_argument("--device", default=-1, type=int, required=False, help="GPU device")
    parser.add_argument("--cache_dir", default=None, type=str, required=False, help="where the model is cached")
    args = parser.parse_args()
    logger.info(args)
    task = 'brainteaser'
    out_dir = os.path.join(args.out_dir)
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(out_dir))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_file = os.path.join(out_dir, 'predictions.jsonl')
    log_file = os.path.join(out_dir, 'results.txt')

    # Load the language model
    device = torch.device(f'cuda:{args.device}') if args.device >= 0 else torch.device("cpu")
    model, tokenizer = init_model(args.lm, device, args.cache_dir)

    # Load the dataset
    instance_reader = BrainteaserInstanceReader()
    
    gold = []
    predictions = []
    results = []
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    print ('currently evaluating the task', task)
    # Predict instances
    sample_id = 0
    with open(out_file, "w") as f_out:
        with open(args.dataset_file) as f_in:
            data = json.load(f_in)
            for fields in tqdm.tqdm(data):
                # print(fields)
                context, question, label, choices = \
                    instance_reader.fields_to_instance(fields)
                # print("context: ", context)
                # print("question: ", question)
                # print("label:", label)
                # print("choices: ", choices)
                gold.append(label)
                if sample_id == 0:
                    results.append(json.dumps(context))
                    results.append(json.dumps(question))
                    results.append(json.dumps(choices))
                prediction = score_task(question, choices, tokenizer, device, model)
                fields["prediction"] = prediction
                fields["correct"] = 1 if prediction == label else 0
                if 'SR' in fields['id']:
                    fields['group'] = 'SR'
                if 'CR' in fields['id']:
                    fields['group'] = 'CR'
                else:
                    fields['group'] = 'OG'
                predictions.append(prediction)
                f_out.write(json.dumps(fields) + "\n")
                sample_id += 1
    # Don't report accuracy if we don't have the labels
    if None not in gold:
        accuracy = (np.array(gold)==np.array(predictions)).mean()
        print(f"Accuracy: {accuracy:.3f}")
        results.append(f"Accuracy: {accuracy:.3f}")
    with open(log_file, 'w') as fout:
        for line in results:
            fout.write(line + '\n')

def get_lm_score(model, batch, label_ids, attention_mask):
    """
    Get the cross entropy loss of the texts in batch using the langage model
    """
    # Batch: [num_choices, max_length]
    with torch.no_grad():
        num_choices, max_length = batch.shape
        label_ids = label_ids.view(-1)
        lm_logits = model(batch, attention_mask=attention_mask)[0]
        lm_logits = lm_logits.view(-1, lm_logits.size(-1))
        loss_fct = CrossEntropyLoss(reduction="none")
        loss = loss_fct(lm_logits, label_ids)
        loss = loss.view(num_choices, -1).sum(1).cpu().numpy()
    return loss


def init_model(model_name: str,
               device: torch.device, cache_dir):
    """
    Initialize a pre-trained LM
    :param model_name: from MODEL_CLASSES
    :param device: CUDA / CPU device
    :return: the model and tokenizer
    """
    logger.info(f'Initializing {model_name}')
    tokenizer = RobertaTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = RobertaForMaskedLM.from_pretrained(model_name, cache_dir=cache_dir)
    model.to(device)
    model.eval()
    return model, tokenizer

if __name__ == '__main__':
    main()