import cv2
import numpy as np
from matplotlib import pyplot as plt


def dataloader_images(data_loader, label_colors, num_images_to_visualize=3):
    count = 0

    fig, axes = plt.subplots(1, num_images_to_visualize, figsize=(15, 5))

    for image, target in data_loader:
        image = np.transpose(np.asarray(image[0]), (1, 2, 0))
        image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        bboxes = np.array(target[0]['boxes'])
        labels = np.array(target[0]['labels'])

        for i, box in enumerate(bboxes):
            category = int(labels[i])
            color = label_colors.get(category, (0, 0, 0))
            cv2.rectangle(image,
                          (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          color, 2)

        axes[count].imshow(image)
        axes[count].axis('off')
        count += 1

        if count == num_images_to_visualize:
            break

    plt.show()


def df_image(df, label_colors):
    # Generate a random index within the range of the DataFrame
    idx = np.random.randint(0, len(df))

    image = cv2.imread(df.iloc[idx]['img_path'])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Retrieve bounding boxes and labels
    bboxes = np.array(df.iloc[idx]['bboxes'])
    labels = np.array(df.iloc[idx]['labels'])

    # Draw bounding boxes on the image
    for i, box in enumerate(bboxes):
        category = int(labels[i])
        color = label_colors.get(category, (0, 0, 0))
        cv2.rectangle(image,
                      (int(box[0]), int(box[1])),
                      (int(box[2]), int(box[3])),
                      color, 2)

    # Display the image
    plt.figure(figsize=(20,10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()
