import numpy as np
import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


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



def collote_fn(batch_samples,tokenizer):
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
test_data = MRPC('valid')