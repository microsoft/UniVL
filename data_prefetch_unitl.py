# pip install prefetch_generator

from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):
    def __iter__(self):
        # transforms generator into a background-thead generator.
        return BackgroundGenerator(super().__iter__(), max_prefetch=1)