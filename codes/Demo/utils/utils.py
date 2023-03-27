from PIL.Image import new
import matplotlib.pyplot as plt
import cv2
import numpy as np

# BGR form of colors
MY_COLORS = [(0, 0, 0), # black
             (0, 120, 0),
             (0, 0, 120),
             (120, 0, 0),
             (0, 255, 255),
             (135, 138, 128), # grey
             (105, 128, 112),
             (215, 235, 250),
             (205, 235, 255),
             (255, 255, 240),
             (87, 207, 227),
             (18, 153, 255),
             (225, 105, 65),
             (205, 90, 106),
             (15, 94, 56),
             (84, 46, 8),
             (87, 201, 0),
             (42, 42, 128),
             (30, 105, 210),
             (80, 127, 255),
             (203, 192, 255),
             (171, 89, 61),
             (201, 161, 51),
             (221, 160, 221)]


def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()

def mask_to_color(mask):
    new_mask = np.zeros([*mask.shape, 3])
    for i in range(len(MY_COLORS)):
        new_mask[mask == i] = MY_COLORS[i]
    return new_mask

def crop_images(image):
    image_original_size = 572
    image_target_size = 388
    img = image[:, (image_original_size-image_target_size)//2: image_target_size+(image_original_size-image_target_size)//2, (image_original_size-image_target_size)//2: image_target_size+(image_original_size-image_target_size)//2]
    return img


