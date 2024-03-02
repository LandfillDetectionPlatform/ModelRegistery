# custom_dataset.py
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class CustomDataset(Dataset):
    def _init_(self, data_dir, transform=None, train=True, test_split=0.2):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # Load images from subdirectories
        for label in ['0', '1']:
            label_dir = os.path.join(data_dir, label)
            for img_name in os.listdir(label_dir):
                self.images.append(os.path.join(label_dir, img_name))
                self.labels.append(int(label))

        # Split data
        if train:
            self.images = self.images[:int(len(self.images) * (1 - test_split))]
            self.labels = self.labels[:int(len(self.labels) * (1 - test_split))]
        else:
            self.images = self.images[int(len(self.images) * (1 - test_split)):]
            self.labels = self.labels[int(len(self.labels) * (1 - test_split)):]

    def _len_(self):
        return len(self.images)

    def _getitem_(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label