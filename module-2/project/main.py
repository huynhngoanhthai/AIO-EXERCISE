import cv2
import numpy as np


def distance(x, y):
    return abs(x - y)


def pixel_wise_matching(left_img, right_img, disparity_range, save_result=True):
    left = cv2.imread(left_img, 0)
    right = cv2.imread(right_img, 0)

    left = left.astype(np.float32)
    right = right.astype(np.float32)

    height, width = left.shape[:2]

    depth = np.zeros((height, width), np.uint8)
    scale = 16
    max_value = 255

    for y in range(height):
        for x in range(width):
            disparity = 0
            cost_min = max_value

            for j in range(disparity_range):
                cost = max_value if (
                    x - j) < 0 else distance(int(left[y, x]), int(right[y, x - j]))

                if cost < cost_min:
                    cost_min = cost
                    disparity = j

            depth[y, x] = disparity * scale

    if save_result:
        print('Saving result...')
        cv2.imwrite('pixel_wise_l1.png', depth)
        cv2.imwrite('pixel_wise_l1_color.png',
                    cv2.applyColorMap(depth, cv2.COLORMAP_JET))

        print('Done.')

    return depth


if __name__ == "__main__":
    pixel_wise_matching("./tsukuba/left.png", "./tsukuba/right.png", 16)
