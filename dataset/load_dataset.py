import torch

from options import Parser
from .aligned_dataset import AlignedDataset


def create_dataset(options: Parser):
    dataset = AlignedDataset()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(options)
    return dataset


class DatasetLoader:
    def __init__(self, options: Parser):
        self.options = options
        self.dataloader = None
        self.dataset = None

    @staticmethod
    def name():
        return 'CustomDatasetDataLoader'

    def initialize(self):
        self.dataset = create_dataset(self.options)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.options.batchSize,
            sampler=data_sampler(self.dataset,
                                 not self.options.serial_batches,
                                 self.options.distributed),
            num_workers=int(self.options.nThreads),
            pin_memory=True)

    @staticmethod
    def load_data():
        return None

    def get_loader(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.options.max_dataset_size)


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return torch.utils.data.RandomSampler(dataset)

    else:
        return torch.utils.data.SequentialSampler(dataset)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch
