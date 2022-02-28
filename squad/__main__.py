import shutil
import click
import json
import random
import os
import pytorch_lightning as pl
import torch

from shutil import copyfile
from more_itertools import flatten

from squad.constants import DATASET_PATHS, CHECKPOINT_PATH, DATASET_ORIGINAL_FORMAT_PATHS
from squad.dataloading import create_training_dataloader, create_inference_dataloader
from squad.qa_model import QAModel


@click.group()
def squad():
    pass


@squad.command()
@click.argument("dev_dataset_file", type=click.File("r"))
def create_datasets(dev_dataset_file):
    original_dev = json.load(dev_dataset_file)
    original_dev_data = original_dev["data"]

    random.seed(1234)
    random.shuffle(original_dev_data)

    os.makedirs("files", exist_ok=True)
    dev_data, test_data, train_data = original_dev_data[:5], original_dev_data[5:10], original_dev_data[10:]

    def flatten_data(data):
        new_data = []
        for doc in data:
            for paragraph in doc["paragraphs"]:
                for question in paragraph["qas"]:
                    new_data.append({
                        "id": question["id"],
                        "context": paragraph["context"],
                        "question": question["question"],
                        "answers": question["answers"],
                        "is_impossible": question["is_impossible"]
                    })
        return new_data

    with open(DATASET_PATHS["train"], "w") as f:
        json.dump(flatten_data(train_data), f)

    with open(DATASET_ORIGINAL_FORMAT_PATHS["train"], "w") as f:
        json.dump({"version": original_dev["version"], "data": train_data}, f)

    with open(DATASET_PATHS["dev"], "w") as f:
        json.dump(flatten_data(dev_data), f)

    with open(DATASET_ORIGINAL_FORMAT_PATHS["dev"], "w") as f:
        json.dump({"version": original_dev["version"], "data": dev_data}, f)

    with open(DATASET_PATHS["test"], "w") as f:
        json.dump(flatten_data(test_data), f)

    with open(DATASET_ORIGINAL_FORMAT_PATHS["test"], "w") as f:
        json.dump({"version": original_dev["version"], "data": test_data}, f)


@squad.command()
@click.option("--learning-rate", type=float, default=0.00005)
@click.option("--batch-size", type=int, default=2)
@click.option("--accu-grads", type=int, default=32)
def train(learning_rate, batch_size, accu_grads):
    with open(DATASET_PATHS["train"]) as f:
        train_data = json.load(f)

    with open(DATASET_PATHS["dev"]) as f:
        dev_data = json.load(f)

    model = QAModel(learning_rate)
    train_dataloader = create_training_dataloader(train_data, shuffle=True, tokenizer=model.tokenizer,
                                                  batch_size=batch_size)
    dev_dataloader = create_training_dataloader(dev_data, shuffle=False, tokenizer=model.tokenizer,
                                                batch_size=batch_size)

    os.makedirs("checkpoints", exist_ok=True)
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        callbacks=[
            pl.callbacks.ModelCheckpoint(dirpath="checkpoints/", monitor="dev_loss", mode="min"),
            pl.callbacks.EarlyStopping(monitor="dev_loss", mode="min", patience=5),
        ],
        accumulate_grad_batches=accu_grads,
        precision=16
    )
    trainer.fit(model, train_dataloader, dev_dataloader)
    copyfile(trainer.checkpoint_callback.best_model_path, CHECKPOINT_PATH)
    shutil.rmtree("checkpoints")


@squad.command()
@click.argument("dataset_split", type=click.Choice(["train", "dev", "test"]))
@click.argument("pred_save_file", type=click.File("w"))
@click.option("--batch-size", type=int, default=2)
def predict(dataset_split, pred_save_file, batch_size):
    with open(DATASET_PATHS[dataset_split]) as f:
        pred_data = json.load(f)

    model = QAModel.load_from_checkpoint(CHECKPOINT_PATH)
    dataloader = create_inference_dataloader(pred_data, model.tokenizer, batch_size=batch_size)
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        precision=16 if torch.cuda.is_available() else 32
    )

    ids = [ex['id'] for ex in pred_data]
    answers = flatten(trainer.predict(model, dataloader))
    predictions = {}
    for idx, answer in zip(ids, answers):
        predictions[idx] = answer

    json.dump(predictions, pred_save_file)
