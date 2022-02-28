import pytorch_lightning as pl
import torch
import transformers

from torch import nn


class QAModel(pl.LightningModule):
    def __init__(self, learning_rate):
        super().__init__()
        pl.seed_everything(1234)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
        self._longformer = transformers.LongformerModel.from_pretrained("allenai/longformer-base-4096")
        self._learning_rate = learning_rate
        self._start_head = nn.Linear(768, 1)
        self._end_head = nn.Linear(768, 1)
        self._loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, batch):
        hidden = self._longformer(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
        )[0]
        start_logits = self._start_head(hidden).squeeze(-1)
        end_logits = self._end_head(hidden).squeeze(-1)
        return start_logits, end_logits

    def training_step(self, batch, batch_idx):
        start_logits, end_logits = self(batch)
        loss = self._loss(start_logits, batch["start_targets"]) + self._loss(end_logits, batch["end_targets"])
        self.log(
            "train_loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["input_ids"].shape[0],
        )
        return loss

    def validation_step(self, batch, batch_idx):
        start_logits, end_logits = self(batch)
        loss = self._loss(start_logits, batch["start_targets"]) + self._loss(end_logits, batch["end_targets"])
        self.log(
            "dev_loss",
            loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["input_ids"].shape[0],
        )
        return loss

    def predict_step(self, batch, batch_idx):
        start_logits, end_logits = self(batch)
        start_positions = start_logits.argmax(dim=-1).tolist()
        end_position = end_logits.argmax(dim=-1).tolist()
        input_ids = batch['input_ids'].tolist()
        answers = []
        for i, positions in enumerate(zip(start_positions, end_position)):
            if positions[0] > positions[1]:
                answers.append("")
            else:
                answer_ids = input_ids[i][positions[0]: positions[1] + 1]
                answers.append(" ".join(self.tokenizer.decode(answer_ids, skip_special_tokens=True).split()))
        return answers

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._learning_rate)
