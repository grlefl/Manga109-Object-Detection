import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from visualize_data import draw_bounding_boxes


# test the model with visual analysis and IoU metric
def test_model(test_df, model, device, label_colors):
    samples = test_df.sample(n=3, replace=False)    # random sample of images from dataframe
    img_paths = samples['img_path'].tolist()

    # define the detection threshold... any detection having score below this will be discarded
    detection_threshold = 0.8

    for i in range(len(img_paths)):

        image = cv2.imread(img_paths[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)                  # convert BGR image to RGB format
        ground_truth = image.copy()
        prediction = image.copy()

        image = image.astype(np.float32) / 255.                         # normalize image for pixel range [0, 1]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)       # bring color channels to front
        image = torch.tensor(image, dtype=torch.float).cuda()           # convert to tensor
        image = torch.unsqueeze(image, 0)                               # add batch dimension

        # pass image through model
        with torch.no_grad():
            outputs = model(image.to(device))

        # load all detection to CPU for further operations
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

        # carry further only if there are detected boxes
        if len(outputs[0]['boxes']) != 0:
            true_bboxes = samples.iloc[i]['bboxes']  # list of lists
            true_labels = samples.iloc[i]['labels']  # list
            pred_bboxes = outputs[0]['boxes'].data.numpy()
            pred_labels = outputs[0]['labels'].data.numpy()
            pred_scores = outputs[0]['scores'].data.numpy()

            # filter out boxes according to `detection_threshold`
            pred_bboxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)

            # draw and display images and bounding boxes
            draw_bounding_boxes(ground_truth, true_bboxes, true_labels, label_colors)       # draw original bboxes
            draw_bounding_boxes(prediction, pred_bboxes, pred_labels, label_colors)         # draw predicted bboxes
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            axes[0].imshow(ground_truth)
            axes[0].set_title('Ground Truth')
            axes[0].axis('off')
            axes[1].imshow(prediction)
            axes[1].set_title('Prediction')
            axes[1].axis('off')
            plt.show()

            print("Number of original bboxes: ", len(true_bboxes))      # original number of bounding boxes
            print("Number of predicted bboxes: ", len(pred_bboxes))     # predicted number of bounding boxes

            iou_counter = 1
            # print out the IoU metric for all boxes that considerably overlap
            for pred_bbox in pred_bboxes:
                for true_bbox in true_bboxes:
                    iou_score = calculate_iou(pred_bbox, true_bbox)
                    if iou_score > 0.5:
                        print("IoU score {}: {}".format(iou_counter, iou_score))
                        iou_counter += 1  # increment counter variable

    print('TEST PREDICTIONS COMPLETE')


# calculate the IoU metric of two boxes
def calculate_iou(box1, box2):
    # calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # calculate area of each box
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # calculate union area
    union_area = area1 + area2 - intersection_area

    # calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0

    return iou
