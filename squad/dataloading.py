import torch
import os

from functools import partial
from torch.utils.data import DataLoader
from collections import Counter


def create_training_dataloader(data, shuffle, tokenizer, batch_size):
    return DataLoader(
        data,
        shuffle=shuffle,
        batch_size=batch_size,
        collate_fn=partial(_collate_fn, tokenizer=tokenizer, input_only=False),
        drop_last=False,
        num_workers=os.cpu_count(),
    )


def create_inference_dataloader(data, tokenizer, batch_size):
    return DataLoader(
        data,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=partial(_collate_fn, tokenizer=tokenizer, input_only=True),
        drop_last=False,
        num_workers=os.cpu_count(),
    )


def _collate_fn(examples, tokenizer, input_only=False):
    contexts = []
    questions = []
    chosen_answers = []
    for e in examples:
        contexts.append(e["context"])
        questions.append(e["question"])
        if not input_only:
            if e["is_impossible"]:
                chosen_answers.append(None)
            else:
                chosen_answers.append(_select_answer(e["answers"]))

    tok_out = tokenizer(contexts, questions, padding=True, return_offsets_mapping=True, return_token_type_ids=True,
                        return_tensors="pt")

    prepared_batch = {
        "input_ids": tok_out.input_ids,
        "attention_mask": tok_out.attention_mask,
        "token_type_ids": tok_out.token_type_ids
    }

    if not input_only:
        start_ids = []
        end_ids = []
        offset_mapping = tok_out.offset_mapping.tolist()
        for i in range(len(offset_mapping)):
            start_idx = None
            end_idx = None
            if chosen_answers[i] is None:
                start_ids.append(0)
                end_ids.append(0)
                continue
            start_char = chosen_answers[i]["answer_start"]
            end_char = chosen_answers[i]["answer_start"] + len(chosen_answers[i]["text"]) - 1
            for j in range(len(offset_mapping[i])):
                if offset_mapping[i][j][0] == offset_mapping[i][j][1] == 0:
                    continue
                if start_idx is None and start_char in range(offset_mapping[i][j][0], offset_mapping[i][j][1]):
                    start_idx = j
                if end_idx is None and end_char in range(offset_mapping[i][j][0], offset_mapping[i][j][1]):
                    end_idx = j

            assert start_idx is not None
            assert end_idx is not None

            start_ids.append(start_idx)
            end_ids.append(end_idx)

        prepared_batch["start_targets"] = torch.tensor(start_ids)
        prepared_batch["end_targets"] = torch.tensor(end_ids)

    return prepared_batch


def _select_answer(answers):
    answers_text = [a["text"] for a in answers]
    answers_freq = Counter(answers_text)
    answers_meta = [(a, answers_freq[a], len(a)) for a in answers_text]
    max_freq = max(c[1] for c in answers_meta)
    chosen_answer_text = min([c for c in answers_meta if c[1] == max_freq], key=lambda c: c[2])[0]
    chosen_answer = None
    for a in answers:
        if a["text"] == chosen_answer_text:
            chosen_answer = a
            break
    assert chosen_answer is not None
    return chosen_answer
