from typing import List, Tuple

from astropy.io import fits
import numpy as np
import cv2

import os


def read_video(input_path: str) -> List[np.ndarray]:
    imgs: List[np.ndarray] = []
    cap: cv2.VideoCapture = cv2.VideoCapture(input_path)

    while cap.isOpened():
        t: Tuple[bool, np.ndarray] = cap.read()
        ret: bool = t[0]
        frame: np.ndarray = t[1]
        if not ret:
            break
        imgs.append(frame)

    cap.release()
    cv2.destroyAllWindows()
    return imgs


def find_contour(img, lower, upper):
    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    # # find the colors within the specified boundaries and apply
    # # the mask
    mask = cv2.inRange(img, lower, upper)
    output = cv2.bitwise_and(img, img, mask=mask)

    ret, thresh = cv2.threshold(mask, 40, 255, 0)
    if (int(cv2.__version__[0]) > 3):
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    else:
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        # draw in blue the contours that were founded
        cv2.drawContours(output, contours, -1, 255, 3)

        # find the biggest countour (c) by the area
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        # draw the biggest contour (c) in green
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return x + 0.5 * w, y + 0.5 * h
    return 0, 0


def calculate_transform_matrix(source, destination):
    src_middle = find_contour(source, [190, 160, 70], [210, 190, 130])  # fluo
    dst_middle = find_contour(destination, [20, 50, 80], [70, 140, 230])  # video
    if src_middle == (0, 0) or dst_middle == (0, 0):
        return np.float32([[1, 0, 0], [0, 1, 0]])
    x = dst_middle[0] - src_middle[0]
    y = dst_middle[1] - src_middle[1]
    return np.float32([[1, 0, x], [0, 1, y]])


if __name__ == '__main__':
    file_dir: str = os.path.dirname(os.path.realpath(__file__))
    video_dir: str = os.path.join(file_dir, 'Movie human colon xenograft 1.mp4')
    fluo_dir: str = os.path.join(file_dir, 'Mice2_cetu2_131213_210250_fluo.fits')
    fluo_data_list: fits.HDUList = fits.open(fluo_dir)
    fluo_images = [img.data for img in fluo_data_list]
    fluo_images.pop(0)
    fluo_data_list.close()
    fluo_images = [cv2.resize(img, (632, 533), interpolation=cv2.INTER_AREA) for img in fluo_images]
    fluo_images = [cv2.flip(img, 1) for img in fluo_images]
    fluo_images = [cv2.threshold(img, 2000, 2500, cv2.THRESH_BINARY)[1] for img in fluo_images]
    kernel = np.ones((5, 5), np.float32) / 25
    fluo_images = [cv2.filter2D(img, -1, kernel) for img in fluo_images]
    fluo_images = [cv2.threshold(img, 850, 2500, cv2.THRESH_BINARY)[1] for img in fluo_images]
    fluo_images = [cv2.applyColorMap(cv2.cvtColor(img,
                                                  cv2.COLOR_GRAY2RGB).astype(np.uint8),
                                     cv2.COLORMAP_OCEAN) for img in fluo_images]

    video_images = read_video(video_dir)
    video_images = [img[27:560, 86:718] for img in video_images]

    fluo_extended = []
    for i in range(len(video_images)):
        idx = int(i / 3.5)
        fluo_extended.append(fluo_images[idx])

    for i in range(len(fluo_extended)):
        M = calculate_transform_matrix(fluo_extended[i], video_images[i])
        fluo_extended[i] = cv2.warpAffine(fluo_extended[i], M, (632, 533))

    mixxed = []
    for i in range(len(fluo_extended)):
        img = cv2.addWeighted(video_images[i], 1.0, fluo_extended[i], 0.6, 0)
        mixxed.append(img)

    writer = cv2.VideoWriter(os.path.join(file_dir, 'mixxed_video_and_fluo_warped.mp4'), -1, 20.0, (632, 533))

    for i in mixxed:
        writer.write(i)
    cv2.waitKey(0)
