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

    # Save grayscale disparity map
    cv2.imwrite(f'{filename}_grayscale.png', disparity_map)

    # Normalize the disparity map for color map visualization
    norm_disparity_map = cv2.normalize(
        disparity_map, None, 0, 255, cv2.NORM_MINMAX)
    norm_disparity_map = np.uint8(norm_disparity_map)

    # Apply a color map
    color_disparity_map = cv2.applyColorMap(
        norm_disparity_map, cv2.COLORMAP_JET)
    cv2.imwrite(f'{filename}_colormap.png', color_disparity_map)


def convert_to_grayscale(left_img, right_img):
    left = cv2.imread(left_img, 0)
    right = cv2.imread(right_img, 0)

    left = left.astype(np.float32)
    right = right.astype(np.float32)
    height, width = left.shape[:2]
    return height, width, left, right


def l1_distance(x, y):
    return abs(x - y)


def pixel_wise_matching_l1(left_img, right_img, disparity_range, kernel_size=16, save_result=True):
    height, width, left, right = convert_to_grayscale(left_img, right_img)

    depth = np.zeros((height, width), np.uint8)

    kernel_half = int((kernel_size - 1) / 2)
    scale = 3
    max_value = 255 * 9

    for y in range(kernel_half, height-kernel_half + 1):
        for x in range(kernel_half, width-kernel_half + 1):

            disparity = 0
            cost_min = 65534

            for j in range(disparity_range):
                total = 0
                value = 0

                for v in range(-kernel_half, kernel_half):
                    for u in range(-kernel_half, kernel_half):
                        value = max_value
                        if (x + u - j) >= 0:
                            value = l1_distance(
                                int(left[y + v, x + u]),  int(right[y + v, (x + u) - j]))
                        total += value

                if total < cost_min:
                    cost_min = total
                    disparity = j

            depth[y, x] = disparity * scale
    if save_result == True:
        print('Saving result...')
        # Save results
        cv2.imwrite('window_based_l1.png', depth)
        cv2.imwrite('window_based_l1_color.png',
                    cv2.applyColorMap(depth, cv2.COLORMAP_JET))

    print('Done.')
    return depth


if __name__ == "__main__":
    # pixel_wise_matching("./tsukuba/left.png", "./tsukuba/right.png", 16)
    left_img_path = 'Aloe/Aloe_left_1.png'
    right_img_path = 'Aloe/Aloe_right_2.png'
    disparity_range = 64
    kernel_size = 5

    left = cv2.imread(left_img_path)
    right = cv2.imread(right_img_path)

    depth = pixel_wise_matching_l1(
        left_img_path,
        right_img_path,
        disparity_range,
        kernel_size=kernel_size,
        save_result=True
    )
