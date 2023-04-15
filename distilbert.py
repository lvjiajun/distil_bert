import os
import random

import numpy as np
import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AdamW, get_scheduler, DistilBertConfig
from transformers import AutoConfig
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertForSequenceClassification, \
    DistilBertPreTrainedModel


class DistilBertForPairwiseCLS(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(768, 768)
        self.classifier = nn.Linear(768, 2)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.post_init()

    def forward(self, x):
        output_layer_norm = self.distilbert(**x)
        cls_vectors = output_layer_norm.last_hidden_state[:, 0, :]
        # cls_vectors = self.pre_classifier(cls_vectors)
        # logits = self.classifier(cls_vectors)
        # # logits = self.dropout(cls_vectors)
        return cls_vectors


def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
seed_everything(42)

learning_rate = 1e-5
batch_size = 32
epoch_num = 36

checkpoint = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(checkpoint)
config = DistilBertConfig.from_pretrained(checkpoint)
model = DistilBertForSequenceClassification.from_pretrained(checkpoint, num_labels=2).to(device)


def collote_fn(batch_samples):
    batch_sentence_1, batch_sentence_2 = [], []
    batch_label = []
    for sample in batch_samples:
        batch_sentence_1.append(sample['sent1'])
        batch_sentence_2.append(sample['sent2'])
        batch_label.append(int(sample['label']))
    X = tokenizer(
        batch_sentence_1,
        batch_sentence_2,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    # offset_mapping = X.pop('offset_mapping')
    # sample_mapping = X.pop('overflow_to_sample_mapping')

    y = torch.tensor(batch_label)
    return X, y


class MRPC(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        re_data = []
        dataset = load_dataset("glue", "mrpc")
        for sent_1, sent_2, label in zip(dataset[data_file]['sentence1'], dataset[data_file]['sentence2'],
                                         dataset[data_file]['label']):
            re_data.append({'sent1': sent_1, 'sent2': sent_2, 'label': label})
        return re_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


train_data = MRPC('train')
valid_data = MRPC('test')

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collote_fn)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=collote_fn)


# def train_loop(dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
#     progress_bar = tqdm(range(len(dataloader)))
#     progress_bar.set_description(f'loss: {0:>7f}')
#     finish_step_num = (epoch - 1) * len(dataloader)
#
#     model.train()
#     for step, (X, y) in enumerate(dataloader, start=1):
#         X['labels'] = y
#         X = X.to(device)
#         pred, loss = model(X['input_ids'], X['attention_mask'], X['labels'])
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         lr_scheduler.step()
#
#         total_loss += loss.item()
#         progress_bar.set_description(f'loss: {total_loss / (finish_step_num + step):>7f}')
#         progress_bar.update(1)
#     return total_loss
def train_loop(dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_step_num = (epoch - 1) * len(dataloader)

    model.train()
    for step, (X, y) in enumerate(dataloader, start=1):
        X, y = X.to(device), y.to(device)

        loss = model(input_ids=X['input_ids'], attention_mask=X['attention_mask'], labels=y)['loss']
        print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss / (finish_step_num + step):>7f}')
        progress_bar.update(1)
    return total_loss


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def test_loop(dataloader, model, mode='Test'):
    assert mode in ['Valid', 'Test']
    size = len(dataloader.dataset)
    correct = 0

    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    model.eval()
    eval_loss, eval_accuracy, nb_eval_steps = 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            out_ = model(input_ids=X['input_ids'], attention_mask=X['attention_mask'], labels=y)
            loss = out_['loss']
            logits = out_['logits']
            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = y.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)
    # Report the final accuracy for this validation run.

    avg_val_accuracy = total_eval_accuracy / size
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / size

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))

    return correct


optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num * len(train_dataloader),
)

total_loss = 0.
best_acc = 0.
for t in range(epoch_num):
    print(f"Epoch {t + 1}/{epoch_num}\n-------------------------------")
    # total_loss = train_loop(train_dataloader, model, optimizer, lr_scheduler, t + 1, total_loss)
    valid_acc = test_loop(valid_dataloader, model, mode='Valid')
    if valid_acc > best_acc:
        best_acc = valid_acc
        print('saving new weights...\n')
        torch.save(model.state_dict(), f'epoch_{t + 1}_valid_acc_{(100 * valid_acc):0.1f}_model_weights.bin')

print("Done!")
