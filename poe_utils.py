import json
import pandas as pd
import torch
import torch.nn.functional as F

from collections import defaultdict
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


def brainteaser_loader(path, multiple_choice_prompt):
    examples_bt = []
    uncond_premise = " the answer is:"
    with open(path) as f:
        data = json.load(f)

    for mcq in data:
        qid = mcq['id']
        label = mcq['label']
        premise = ' ' + mcq['question']
        premise = premise[:-1] + '?'

        options = mcq['choice_list']
        if multiple_choice_prompt is not None:
            hypotheses = ["A", "B", "C", "D"]
            # premise = f"{multiple_choice_prompt} Question: {premise}\nA. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}\nE. {options[4]}\nAnswer:"
            premise = f"""{multiple_choice_prompt} Question: {premise}\n\nWhat is the correct answer to the question from the following choices?\n(A) {options[0]}\n(B) {options[1]}\n(C) {options[2]}\n(D) {options[3]}"""
        else:
            hypotheses = options
            premise = premise + uncond_premise
        examples_bt += [{
            'qid': qid,
            'label': label,
            'premise': premise,
            'uncond_premise': uncond_premise,
            'hypothesis0': hypotheses[0],
            'hypothesis1': hypotheses[1],
            'hypothesis2': hypotheses[2],
            'hypothesis3': hypotheses[3],

        }]
    return examples_bt


def load_data(multiple_choice_prompt):
    num_options = 4
    ending_names = ['hypothesis0', 'hypothesis1', 'hypothesis2', 'hypothesis3']
    header_name = 'premise'
    # file_path = "data/SP-eval.json"
    file_path = "data/SP-train.json"
    train_file_path = None
    loader = brainteaser_loader

    dev_data = loader(file_path, multiple_choice_prompt)
    dev_dataset = Dataset.from_list(dev_data).with_format("torch")

    if train_file_path is not None:
        train_data = loader(train_file_path, multiple_choice_prompt)
        train_dataset = Dataset.from_list(train_data).with_format("torch")
    else:  # BB tasks have no train set.
        train_dataset = dev_dataset
    return ending_names, header_name, dev_dataset, train_dataset


def preprocess_function_seq2seq(examples, **kwargs):
    ending_names, header_name, tokenizer = kwargs['ending_names'], kwargs['header_name'], kwargs['tokenizer']
    num_choice = len(ending_names)
    question_headers = examples[header_name]
    # the tokenizer handles multiple spaces.
    first_sentences = [[context] * len(ending_names)
                       for context in examples[header_name]]
    second_sentences = [
        [f"{examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
    ]

    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    # tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
    tokenized_headers = tokenizer(
        first_sentences, padding=True, truncation=True)
    tokenized_endings = tokenizer(
        second_sentences, padding=True, truncation=True)
    header_dict = {f"header_{k}": [v[i: i + num_choice] for i in range(
        0, len(v), num_choice)] for k, v in tokenized_headers.items()}
    ending_dict = {f"ending_{k}": [v[i: i + num_choice] for i in range(
        0, len(v), num_choice)] for k, v in tokenized_endings.items()}
    return {**header_dict, **ending_dict}


def compute_conditional_score_seq2seq(batch, model, device, pad_token_id):
    # returns log_prob of p(y|x) for each batch

    # e.g., (batch_size, #option, ending_seq_len): (32, 2, 18)
    ending_shape = batch["ending_input_ids"].shape
    # flatten. both input_ids has 0 as padding token.
    header_input_ids = batch["header_input_ids"].view(
        -1, batch["header_input_ids"].shape[-1]).to(device)
    header_attention_mask = batch["header_attention_mask"].view(
        -1, batch["header_attention_mask"].shape[-1]).to(device)
    ending_input_ids = batch["ending_input_ids"].view(
        -1, batch["ending_input_ids"].shape[-1]).to(device)

    model = model.to(device)

    # adding this line of code takes me more than an hour.
    # without adding torch.no_grad, GPU usage will muiltply by 4.
    with torch.no_grad():
        outputs = model(input_ids=header_input_ids,
                        attention_mask=header_attention_mask,
                        labels=ending_input_ids)

    _, logits = outputs.loss, outputs.logits
    # e.g., (batch_size * #option, ending_seq_len, #vocab): (64, 18, 32128)
    logits = logits.view(-1, logits.shape[-1])
    # ignore padding token: 0
    ce_loss = F.cross_entropy(logits, ending_input_ids.view(-1),
                              reduction="none", ignore_index=pad_token_id).detach().cpu()
    # each score is the negative log-likelihood of a ending given a header.
    # batch_predictions = ce_loss.view(ending_shape).sum(dim=-1).argmin(dim=-1)
    log_prob = ce_loss.view(ending_shape).sum(dim=-1)
    return log_prob


def inference_language_modeling(model, eval_dataloader, device, compute_func, pad_token_id):
    model.eval()
    lm_predictions = torch.zeros(0)
    avg_lm_predictions = torch.zeros(0)
    labels = torch.zeros(0)
    torch.cuda.empty_cache()
    avg_log_probs = []

    pbar = tqdm(eval_dataloader, desc="Inference")
    for batch in pbar:
        log_prob = compute_func(batch, model, device, pad_token_id)
        avg_log_prob = log_prob / batch["ending_attention_mask"].sum(dim=-1)
        avg_log_probs.append(avg_log_prob)

        batch_predictions = log_prob.argmin(dim=-1)
        batch_avg_predictions = avg_log_prob.argmin(dim=-1)

        batch_labels = batch["label"]
        lm_predictions = torch.cat((lm_predictions, batch_predictions))
        avg_lm_predictions = torch.cat(
            (avg_lm_predictions, batch_avg_predictions))
        labels = torch.cat((labels, batch_labels))

        # make accuracy accumulative
        lm_accuracy = (lm_predictions == labels).sum().item() / len(labels)
        avg_lm_accuracy = (avg_lm_predictions ==
                           labels).sum().item() / len(labels)
        pbar.set_description(
            f"Language modeling accuracy: {lm_accuracy:.4f}, Average language modeling accuracy: {avg_lm_accuracy:.4f}")
    avg_log_probs = torch.cat(avg_log_probs, dim=0)
    return avg_log_probs, lm_accuracy, avg_lm_accuracy


def compute_mask_process_of_elimination(avg_log_probs, mask_strategy, **kwargs):
    masks = torch.ones_like(avg_log_probs)
    if mask_strategy == "lowest":
        # soft masking (v1), i.e., get rid of the least likely answer.
        masks[torch.arange(avg_log_probs.shape[0]),
              avg_log_probs.argmax(dim=-1)] = 0
    elif mask_strategy == "below_average":
        # v2: Calculate the row-wise mean
        row_mean = avg_log_probs.mean(dim=1, keepdim=True)
        # Set values below the mean to 0
        masks[avg_log_probs > row_mean] = 0
    elif mask_strategy == "lowest_iter":
        # similar to lowest, but ignore inf, and mask from the remaining options.
        # soft masking (v1), i.e., get rid of the least likely answer.
        avg_log_probs[avg_log_probs == float("inf")] = float("-inf")
        masks[torch.arange(avg_log_probs.shape[0]),
              avg_log_probs.argmax(dim=-1)] = 0
        # set mask that correspond to inf to 0
        masks[avg_log_probs == float("-inf")] = 0
    elif mask_strategy == "min_k":
        min_k = kwargs["min_k"]
        # keep the min k options
        avg_log_probs_f32 = avg_log_probs.float()
        _, min_k_indices = avg_log_probs_f32.topk(min_k, dim=-1)
        masks[torch.arange(avg_log_probs_f32.shape[0]
                           ).unsqueeze(-1), min_k_indices] = 0
    else:
        raise NotImplementedError
    return masks


def create_multiple_choice_prompt(example, **kwargs):
    alphabets = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    multiple_choice_prompt = kwargs["multiple_choice_prompt"]
    scoring_method = kwargs["scoring_method"]
    num_of_options = kwargs["num_of_options"]
    mask = example['mask']
    # null_string = f"[MASK]"
    if kwargs['mask_token'] is not None:
        null_string = kwargs['mask_token']
    else:
        null_string = f"[MASK]"
    mcp_example = {}
    # example['premise'] = premise = f"{multiple_choice_prompt} {premise}\nA. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}\nE. {options[4]}\nAnswer:"
    # premise = f"{multiple_choice_prompt} Question: {example['premise']}\n"

    if scoring_method != "multiple_choice_prompt":
        premise = f"{multiple_choice_prompt}\n Question: {example['premise']}\n"
        premise = premise.replace(f"{example['uncond_premise']}", "")
        for idx, single_mask in enumerate(mask):
            mcp_example[f'hypothesis{idx}'] = alphabets[idx]
            if single_mask == 1:
                premise += f"{alphabets[idx]}. {example[f'hypothesis{idx}']}\n"
            else:
                # consider other null strings.
                premise += f"{alphabets[idx]}. {null_string}\n"
        premise += "Answer:"
    else:  # for multiple choice prompt, options are already presented in the premise.
        premise = f"{multiple_choice_prompt}\n{example['premise']}"
        premise = premise.replace(f"{example['uncond_premise']}", "")
        for idx, single_mask in enumerate(mask):
            option_start_index = premise.rfind(f"({alphabets[idx]}) ")
            if idx == num_of_options - 1:
                option_end_index = premise.rfind(f"Answer:")
            else:
                option_end_index = premise.rfind(f"({alphabets[idx + 1]}) ")
            option = premise[option_start_index:option_end_index]
            if single_mask == 1:
                pass
            else:
                # consider other null strings.
                premise = premise.replace(
                    option, f"({alphabets[idx]}) {null_string}\n")
    mcp_example['premise'] = premise
    return mcp_example


def inference_process_of_elimination(model, eval_dataloader, device, compute_func, pad_token_id):
    model.eval()
    lm_predictions = torch.zeros(0)
    avg_lm_predictions = torch.zeros(0)
    labels = torch.zeros(0)
    torch.cuda.empty_cache()
    avg_log_probs = []

    pbar = tqdm(eval_dataloader, desc="Inference")
    for batch in pbar:
        log_prob = compute_func(batch, model, device, pad_token_id)
        # apply hard masking
        log_prob[batch["mask"] == 0] = float("inf")
        avg_log_prob = log_prob / batch["ending_attention_mask"].sum(dim=-1)
        avg_log_probs.append(avg_log_prob)

        batch_predictions = log_prob.argmin(dim=-1)
        batch_avg_predictions = avg_log_prob.argmin(dim=-1)

        batch_labels = batch["label"]
        lm_predictions = torch.cat((lm_predictions, batch_predictions))
        avg_lm_predictions = torch.cat(
            (avg_lm_predictions, batch_avg_predictions))
        labels = torch.cat((labels, batch_labels))

        # make accuracy accumulative
        lm_accuracy = (lm_predictions == labels).sum().item() / len(labels)
        avg_lm_accuracy = (avg_lm_predictions ==
                           labels).sum().item() / len(labels)
        pbar.set_description(
            f"Process of elimination accuracy: {lm_accuracy:.4f}, Average process of elimination accuracy: {avg_lm_accuracy:.4f}")
    avg_log_probs = torch.cat(avg_log_probs, dim=0)
    return lm_predictions, lm_accuracy, avg_lm_accuracy


def write_to_csv(path, mcp_dataset, lm_predictions):
    lm_predictions = lm_predictions.tolist()
    predictions_data = defaultdict(list)

    for i, mcq in enumerate(mcp_dataset):
        qid = mcq['qid']
        predictions_data['id'].append(qid)
        predictions_data['question'].append(mcq['premise'])
        gold_label = mcq['label'].item()
        predictions_data['label'].append(gold_label)
        if 'SR' in qid:
            predictions_data['group'].append('SR')
        elif 'CR' in qid:
            predictions_data['group'].append('CR')
        else:
            predictions_data['group'].append('OG')
        pred_label = int(lm_predictions[i])
        predictions_data['prediction'].append(pred_label)
        correct = 1 if pred_label == gold_label else 0
        predictions_data['correct'].append(correct)

    predictions_df = pd.DataFrame(predictions_data)
    predictions_df.to_csv(path)
