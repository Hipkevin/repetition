import pandas as pd
import numpy as np
import gensim
import torch

from torch.utils.data import Dataset


def getVector(text, model, pad_size):
    X = []
    for document in text:
        x = []
        for word in document:
            x.append(np.array(model.wv.get_vector(word)))

        if len(x) > pad_size:
            x = x[:pad_size]
        else:
            x += [np.zeros(300)] * (pad_size - len(x))

        X.append(x)

    return X


def getData(data_path, label_path):
    data = pd.read_excel(data_path)
    label = pd.read_csv(label_path, delimiter=',')

    index = data['id']
    text = data['text']

    data_dict = dict()
    for idx, content in zip(index, text):
        data_dict[idx] = content

    qID = label['questionID']
    vID = label['duplicates']
    tag = label['label']

    q_text = list()
    v_text = list()
    label_list = list()
    for i in range(len(tag)):
        if tag[i] == 0:
            q = eval(data_dict[qID[i]])
            v = eval(data_dict[qID[i-1]])
        else:
            q = eval(data_dict[qID[i]])
            v = eval(data_dict[vID[i]])

        q_text.append(q)
        v_text.append(v)
        label_list.append(tag[i])

    return q_text, v_text, label_list


class RepeatDataset(Dataset):
    def __init__(self, label_path, config):
        super(RepeatDataset, self).__init__()
        data_path = config.data_path
        model_path = config.model_path

        q_text, v_text, label = getData(data_path, label_path)

        model = gensim.models.word2vec.Word2Vec.load(model_path)

        self.q_vec = torch.tensor(getVector(q_text, model, config.pad_size), dtype=torch.float)
        self.v_vec = torch.tensor(getVector(v_text, model, config.pad_size), dtype=torch.float)
        self.label = torch.tensor(label, dtype=torch.float)

    def __getitem__(self, index):
        return self.q_vec[index], self.v_vec[index], self.label[index]

    def __len__(self):
        return len(self.label)