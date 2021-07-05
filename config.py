import torch

class Config:
    def __init__(self):
        super(Config, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.data_path = 'data/data.xlsx'
        self.label_path = 'data/label.csv'
        self.model_path = 'embedding/skip-gram_300'

        self.epoch_size = 3
        self.batch_size = 32
        self.pad_size = 128
        self.learning_rate = 1e-5
        self.weight_decay = 1e-2