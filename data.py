from torch.utils.data import Dataset, DataLoader, Sampler
import torch
import re
from collections import Counter
import numpy as np

def process(token):
    token = token.strip().lower()
    try:
        float(token)
        return "<num>"
    except ValueError:
        return token

class BatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset_len = len(dataset)
        self.batch_size = batch_size
        self.split_len = len(dataset)//batch_size
        self.dataset_len_drop = self.split_len * batch_size

    def __len__(self):
        return self.split_len

    def __iter__(self):
        for i in range(self.split_len):
            yield range(i, self.dataset_len_drop, self.split_len)

class AugmentDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.max_len = max(len(d) for d in datasets)
    
    def __getitem__(self, i):
        return tuple(d[i % len(d)] for d in self.datasets)

    def __len__(self):
        return self.max_len


class tag_dataset(Dataset):
    def __init__(self, data_path, cat_path, seq_len, pred_offset=0, w2idx=None, c2idx=None, cutoff=5, strip_feat=False):
        super(tag_dataset, self).__init__()
        
        if c2idx is None:
            self.categories, self.c2idx = self.load_categories(cat_path, strip_feat)
        else:
            self.c2idx = c2idx
            self.categories = list(c2idx.keys())
        self.data = self.load(data_path, strip_feat)
        if w2idx is None:
            self.vocab, self.w2idx = self.build_vocab(cutoff)
        else:
            self.w2idx = w2idx
            self.vocab = list(w2idx.keys())
        self.indexify()
        self.data = self.split_data(seq_len, pred_offset)

    def load_categories(self, path, strip_feat):
        categories = ["EOS", "_"]
        with open(path) as cat_f:
            for line in cat_f:
                tag = line.strip()
                if strip_feat: tag = re.sub("\[.*\]", "", tag)
                categories.append(tag)

        categories = list(set(categories)) # remove dups if strip_feat=True
        c2idx = dict(zip(categories, range(len(categories))))
        return categories, c2idx

    def load(self, path, strip_feat):
        data = []
        with open(path) as data_f:
            for line in data_f:
                if line.strip() == "":
                    data.append(("<eos>", "EOS"))
                    continue
                token, tag = line.split("\t")
                tag = tag.strip()

                # remove ccg features ([pss], [dcl], etc)
                if strip_feat: tag = re.sub("\[.*\]", "", tag)
                data.append((process(token), tag))
        return data

    def build_vocab(self, cutoff):
        c = Counter([token for token,_ in self.data])
        vocab = [x for x,y in c.items() if y > cutoff] + ["<oov>"]
        w2idx = dict(zip(vocab, range(len(vocab))))
        return vocab, w2idx

    def indexify(self):
        self.data = [(self.w2idx.get(w, self.w2idx["<oov>"]), self.c2idx[t]) for (w, t) in self.data]

    def split_data(self, seq_len, pred_offset):
        out = []

        i = 0
        while i < len(self.data) - 5:
            #bptt = int(max(5, np.random.normal(seq_len, 5)))
            bptt = seq_len
            if bptt > len(self.data) - i - 1:
                break
            out.append((torch.LongTensor([x[0] for x in self.data[i:i+bptt]]), 
                        torch.LongTensor([x[1] for x in self.data[i+pred_offset:i+bptt+pred_offset]]),
                        ))
            i += bptt

        return out

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class tag_lm(tag_dataset):
    def __init__(self, data_path, cat_path, seq_len, w2idx=None, c2idx=None, cutoff=5, strip_feat=False):
        if c2idx is None:
            self.categories, self.c2idx = self.load_categories(cat_path, strip_feat)
        else:
            self.c2idx = c2idx
            self.categories = list(c2idx.keys())
        self.data = self.load(data_path, strip_feat)
        if w2idx is None:
            self.vocab, self.w2idx = self.build_vocab(cutoff)
        else:
            self.w2idx = w2idx
            self.vocab = list(w2idx.keys())
        self.indexify()
        self.data = self.split_data(seq_len)


    def split_data(self, seq_len):
        out = []

        i = 0
        while i < len(self.data) - 5:
            #bptt = int(max(5, np.random.normal(seq_len, 5)))
            bptt = seq_len
            if bptt > len(self.data) - i - 1:
                break
            out.append((torch.LongTensor([x[1] for x in self.data[i:i+bptt]]), 
                        torch.LongTensor([x[1] for x in self.data[i+1:i+bptt+1]]),
                        ))
            i += bptt

        return out



class joint_tag_lm(Dataset):
    def __init__(self, data_path, cat_path, seq_len, w2idx=None, cutoff=5, strip_feat=False):
        super(joint_tag_lm, self).__init__()
        
        self.categories, self.c2idx = self.load_categories(cat_path)
        self.data = self.load(data_path, strip_feat)
        if w2idx == None:
            self.vocab, self.w2idx = self.build_vocab(cutoff)
        else:
            self.w2idx = w2idx
            self.vocab = list(w2idx.keys())
        self.indexify()
        self.data = self.split_data(seq_len)

    def load_categories(self, path):
        categories = ["EOS", "_"]
        with open(path) as cat_f:
            for line in cat_f:
                categories.append(line.strip())
        c2idx = dict(zip(categories, range(len(categories))))
        return categories, c2idx

    def load(self, path, strip_feat=False):
        data = []
        with open(path) as data_f:
            for line in data_f:
                if line.strip() == "":
                    data.append(("<eos>", "EOS"))
                    continue
                token, tag = line.split("\t")
                tag = tag.strip()
                if strip_feat: tag = re.sub("\[\.*\]", "", tag)
                data.append((process(token), tag))
        return data

    def build_vocab(self, cutoff):
        c = Counter([token for token,_ in self.data])
        vocab = [x for x,y in c.items() if y > cutoff] + ["<oov>"]
        w2idx = dict(zip(vocab, range(len(vocab))))
        return vocab, w2idx

    def indexify(self):
        self.data = [(self.w2idx.get(w, self.w2idx["<oov>"]), self.c2idx[t]) for (w, t) in self.data]

    def split_data(self, seq_len):
        out = []

        i = 0
        while i < len(self.data) - 5:
            #bptt = int(max(5, np.random.normal(seq_len, 5)))
            bptt = seq_len
            if bptt > len(self.data) - i - 1:
                break
            out.append((torch.LongTensor([x[0] for x in self.data[i:i+bptt]]), 
                        torch.LongTensor([x[0] for x in self.data[i+1:i+bptt+1]]),
                        torch.LongTensor([x[1] for x in self.data[i:i+bptt]]),
                        ))
            i += bptt

        return out

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class augment_tag_lm(joint_tag_lm):
    def __init__(self, lm_path, tag_path, seq_len, w2idx = None, cutoff=0, strip_feat=False):
        self.data = self.load(lm_path, tag_path, strip_feat)
        if w2idx == None:
            self.vocab, self.w2idx = self.build_vocab(cutoff)
        else:
            self.w2idx = w2idx
            self.vocab = list(w2idx.keys())
        self.indexify()
        self.data = self.split_data(seq_len)

    def load(self, lm_path, tag_path, strip_feat=False):
        lm_data = []
        with open(lm_path, encoding="utf-8") as lm_f:
            for line in lm_f:
                words = line.split()
                lm_data += [process(x) for x in words]
        if tag_path is not None:
            tag_data = super().load(tag_path, strip_feat)
            lm_data += [x for (x, _) in tag_data]
        return lm_data

    def build_vocab(self, cutoff):
        c = Counter([token for token in self.data])
        vocab = [x for x,y in c.items() if y > cutoff] + ["<oov>"]
        w2idx = dict(zip(vocab, range(len(vocab))))
        return vocab, w2idx

    def indexify(self):
        self.data = [self.w2idx.get(w, self.w2idx["<oov>"]) for w in self.data]

    def split_data(self, seq_len):
        out = []

        i = 0
        while i < len(self.data) - 5:
            #bptt = int(max(5, np.random.normal(seq_len, 5)))
            bptt = seq_len
            if bptt > len(self.data) - i - 1:
                break
            out.append((torch.LongTensor([x for x in self.data[i:i+bptt]]), 
                        torch.LongTensor([x for x in self.data[i+1:i+bptt+1]]),
                        ))
            i += bptt

        return out

if __name__ == "__main__":
    data = tag_lm("data/ccg_supertags/ccg.02-21.common", "data/ccg_supertags/categories", 40)
    sampler = BatchSampler(data, 5)
    loader = DataLoader(data, batch_sampler=sampler)

    for input, lm_target in loader:
        for i, sent in enumerate(input):
            print("input data: \t" + " ".join([data.categories[x] for x in sent]))
            print("lm  target: \t" + " ".join([data.categories[x] for x in lm_target[i]]))
            print()
        break


    data = joint_tag_lm("data/ccg_supertags/ccg.02-21.common", "data/ccg_supertags/categories", 40)
    sampler = BatchSampler(data, 10)
    loader = DataLoader(data, batch_sampler=sampler)
    print(len(data) * 40) 
    print(len(loader))

    for j, (input, lm_target, ccg_target) in enumerate(loader):
        for i, sent in enumerate(input):
            print("input data: \t" + " ".join([data.vocab[x] for x in sent]))
            print("ccg target: \t" + " ".join([data.categories[x] for x in ccg_target[i]]))
            print("lm  target: \t" + " ".join([data.vocab[x] for x in lm_target[i]]))
            print()
            if i > 3: break
        print("---")
        if j > 3: break

    data = augment_tag_lm("data/gulordava/train.txt", "data/ccg_supertags/ccg.02-21.common", 40)
    loader = DataLoader(data, batch_size=10, drop_last=True)
    print(len(data) * 40) 
    print(len(loader))

    for input, lm_target in loader:
        for i, sent in enumerate(input):
            print("input data: \t" + " ".join([data.vocab[x] for x in sent]))
            print("lm  target: \t" + " ".join([data.vocab[x] for x in lm_target[i]]))
            print()
        break

    data2 = tag_dataset("data/ccg_supertags/ccg.02-21.common", "data/ccg_supertags/categories", 40, w2idx=data.w2idx)
    merge_data = AugmentDataset(data, data2)
    sampler = BatchSampler(merge_data, 5)
    loader = DataLoader(merge_data, batch_sampler=sampler)

    for j, ((lm_input, lm_target),(ccg_input, ccg_target)) in enumerate(loader):
        for i in range(5):
            print("lm input data: \t" + " ".join([data.vocab[x] for x in lm_input[i]]))
            print("lm  target: \t" + " ".join([data.vocab[x] for x in lm_target[i]]))
            print()
            print("ccg input data: \t" + " ".join([data.vocab[x] for x in ccg_input[i]]))
            print("ccg  target: \t" + " ".join([data2.categories[x] for x in ccg_target[i]]))
            print("\n")
            if i > 3: break
        print("---")
        if j > 3: break

