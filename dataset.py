from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, img_df, transform=None):
        self.img_df = img_df
        self.transform = transform

    def __len__(self):
        return len(self.img_df)

    def __getitem__(self, idx):
        img_path = self.img_df.iloc[idx]['img_path']
        bboxes = self.img_df.iloc[idx]['bboxes']
        labels = self.img_df.iloc[idx]['labels']

        # load the image using PIL
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image, bboxes, labels = self.transform(image=image, bboxes=bboxes, class_labels=labels)

        return image, bboxes, labels
