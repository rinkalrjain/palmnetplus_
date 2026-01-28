from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class PalmprintDataset(Dataset):
    def __init__(self, low_dir, gt_dir):
        self.low = sorted(os.listdir(low_dir))
        self.gt = sorted(os.listdir(gt_dir))
        self.low_dir = low_dir
        self.gt_dir = gt_dir
        self.tf = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((256,256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.low)

    def __getitem__(self, idx):
        return (
            self.tf(Image.open(os.path.join(self.low_dir, self.low[idx]))),
            self.tf(Image.open(os.path.join(self.gt_dir, self.gt[idx])))
        )
