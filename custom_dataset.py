import os 
from PIL import Image
from torch.utils.data import Dataset


class ImageNet1K(Dataset):
    def __init__(self, image_path, labels, transform=None):
        self.image_path = image_path # path
        self.labels = list(labels.keys()) # dict
        self.transform = transform

        self.image_files = [os.path.join(self.image_path, img) for img in os.listdir(self.image_path)]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_pil = Image.open(self.image_files[idx]).convert('RGB')
        label_str = self.image_files[idx].split('/')[-1].split('_')[-1].strip('.JPEG')
        label = self.labels.index(label_str)

        if self.transform:
            image = self.transform(image_pil, return_tensors="pt")

        return image, label