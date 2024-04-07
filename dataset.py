import os
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import pil_loader


class CustomDataset(Dataset):       # taken from github!!!
    # initialize configurations
    def __init__(self, images, width, height, classes, transforms=None):

        self.transforms = transforms
        self.images = images
        self.height = height
        self.width = width
        self.classes = classes

        self.image_paths = images["img_path"].to_list()
        self.image_annotations = images["page_annotation"]
        self.all_images = ["".join(path.split(os.path.sep)[-2:]) for path in self.image_paths] # okay for now

    def __len__(self):
        return len(self.all_images)

    # get (image, target)
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = pil_loader(image_path)                              # read image
        image_resized = image.resize((self.height, self.width))     # resize image

        boxes = []
        labels = []
        #authors = []

        image_width, image_height = image.size

        for annotation_type in self.classes[1:]:
            # extract all annotations of the current class
            rois = self.image_annotations[idx][annotation_type]
            for roi in rois:
                labels.append(self.classes.index(annotation_type))
                #authors.append(roi["author"])
                # xmin = left corner x-coordinates
                xmin = roi["@xmin"]
                # xmax = right corner x-coordinates
                xmax = roi["@xmax"]
                # ymin = left corner y-coordinates
                ymin = roi["@ymin"]
                # ymax = right corner y-coordinates
                ymax = roi["@ymax"]

                # resize bounding box according to the desired size
                xmin_final = (xmin/image_width)*self.width
                xmax_final = (xmax/image_width)*self.width
                ymin_final = (ymin/image_height)*self.height
                yamx_final = (ymax/image_height)*self.height

                boxes.append([xmin_final, ymin_final, xmax_final, yamx_final])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)                 # bounding box to tensor
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])    # area of the bounding boxes
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)         # no crowd instances
        labels = torch.as_tensor(labels, dtype=torch.int64)                 # labels to tensor
        #authors = torch.as_tensor(authors, dtype=torch.int64)

        # prepare the final dictionary
        target = {"boxes": boxes, "labels": labels, "area": area, "author": authors, "iscrowd": iscrowd}
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        # apply the image transforms
        if self.transforms:
            image_resized, target = self.transforms(image_resized, target)

        return image_resized, target
