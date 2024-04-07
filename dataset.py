import cv2
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, img_df, transform=None):
        self.img_df = img_df
        self.transform = transform

    def __len__(self):
        return len(self.img_df)

    def __getitem__(self, idx):
        img_path = self.img_df.iloc[idx]['img_path']
        bboxes = self.img_df.iloc[idx]['bboxes']        # [[box#1], [box#2], [box#3]]
        labels = self.img_df.iloc[idx]['labels']        # [label#1, label#2, label#3]

        image = cv2.imread(img_path)                            # load the image as numpy array (BGR format)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)          # convert BGR image to RGB

        if self.transform:
            transformed = self.transform(image=image, bboxes=bboxes, labels=labels)
            image = transformed['image']
            bboxes = transformed['bboxes']
            labels = transformed['labels']

        return image, bboxes, labels
