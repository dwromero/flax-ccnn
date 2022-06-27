import numpy as np

# config
from hydra import utils


class DataModule:
    def __init__(
        self,
        data_dir,
        batch_size,
        test_batch_size,
        num_workers,
    ):
        super().__init__()

        # add values to self
        self.data_dir = utils.get_original_cwd() + data_dir
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

    def construct_dataloaders(self):
        self.train_dataloader = self.train_dataloader()
        self.val_dataloader = self.val_dataloader()
        self.test_dataloader = self.test_dataloader()

    def prepare_data(self):
        raise NotImplementedError

    def setup(self):
        raise NotImplementedError

    def train_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError

    @staticmethod
    def default_collate_fn(batch) -> tuple[np.array, np.array]:
        """Provides us with batches of numpy arrays."""
        transposed_data = list(zip(*batch))
        imgs = np.stack(transposed_data[0])
        labels = np.array(transposed_data[1])
        return imgs, labels

    @staticmethod
    # Transformations applied on each image => bring them into a numpy array
    def image_to_numpy(img, data_mean, data_std):
        img = np.array(img, dtype=np.float32)
        img = (img / 255. - data_mean) / data_std
        return img



