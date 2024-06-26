import cv2
import numpy as np
from matplotlib import pyplot as plt


# simple function to draw bounding boxes on an image
# assuming pascal_voc formatting
def draw_bounding_boxes(image, bboxes, labels, label_colors):
    for i, box in enumerate(bboxes):
        category = int(labels[i])
        color = label_colors.get(category, (0, 0, 0))  # default color is black
        cv2.rectangle(image,
                      (int(box[0]), int(box[1])),
                      (int(box[2]), int(box[3])),
                      color, 2)
    return image


# take a random image from the dataframe display it with bounding boxes
def df_image(df, label_colors):
    # generate a random index within the range of the DataFrame
    idx = np.random.randint(0, len(df))

    image = cv2.imread(df.iloc[idx]['img_path'])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # retrieve bounding boxes and labels
    bboxes = np.array(df.iloc[idx]['bboxes'])
    labels = np.array(df.iloc[idx]['labels'])

    # draw bounding boxes on the image
    draw_bounding_boxes(image, bboxes, labels, label_colors)

    # display the image
    plt.figure(figsize=(20,10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()


# display a number of images and their bounding boxes from a dataloader
def dataloader_images(data_loader, label_colors, num_images_to_visualize=3):
    count = 0

    fig, axes = plt.subplots(1, num_images_to_visualize, figsize=(15, 5))

    # retrieve image from dataloader, revert it back to normal
    for image, target in data_loader:
        image = np.transpose(np.asarray(image[0]), (1, 2, 0))
        image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        bboxes = np.array(target[0]['boxes'])
        labels = np.array(target[0]['labels'])

        draw_bounding_boxes(image, bboxes, labels, label_colors)

        axes[count].imshow(image)
        axes[count].axis('off')
        count += 1

        if count == num_images_to_visualize:
            break

    plt.show()


# display training and validation loss plots
def results_images(image1_path, image2_path, title1='Training Loss', title2='Validation Loss'):
    # load images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    # display images
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].imshow(image1)
    axes[0].set_title(title1)
    axes[1].imshow(image2)
    axes[1].set_title(title2)
    plt.show()
