import cv2
import numpy as np
from matplotlib import pyplot as plt


def visualize_images(data_loader, label_colors, num_images_to_visualize=3):
    count = 0

    fig, axes = plt.subplots(1, num_images_to_visualize, figsize=(15, 5))

    for image, target in data_loader:
        image = np.transpose(np.asarray(image[1]), (1, 2, 0))
        image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        bboxes = np.array(target[1]['boxes'])
        labels = np.array(target[1]['labels'])

        for i, box in enumerate(bboxes):
            class_name = str(labels[i])
            color = label_colors.get(class_name, (255, 255, 255))
            cv2.rectangle(image,
                          (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          color, 2)
            cv2.putText(image, class_name,
                        (int(box[0]), int(box[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color,
                        2, lineType=cv2.LINE_AA)

        axes[count].imshow(image)
        axes[count].axis('off')
        count += 1

        if count == num_images_to_visualize:
            break

    plt.show()
