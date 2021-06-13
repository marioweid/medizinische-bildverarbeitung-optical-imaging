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
    # video shape=(576,768) len=4896
    # fluo  shape=(256,256) len=1405

    fluo_extended = []
    for i in range(len(video_images)):
        idx = int(i / 3.5)
        fluo_extended.append(fluo_images[idx])

    mixxed = []
    for i in range(len(fluo_extended)):
        img = cv2.addWeighted(video_images[i], 1.0, fluo_extended[i], 0.6, 0)
        mixxed.append(img)

    writer = cv2.VideoWriter(os.path.join(file_dir, 'mixxed_video_and_fluo.mp4'), -1, 20.0, (632, 533))

    for i in mixxed:
        writer.write(i)
    cv2.waitKey(0)

    # left = 86
    # top = 27  # x1 = (86,27)
    # bot = 560  # x2 = (718,560)
    # right = 718
    # mixed_cropped = mixed_img[top:bot, left:right]
    # cv2.imshow('mixxed_image', mixed_img)
    # cv2.imshow('mixxed_cropped', mixed_cropped)

    # VSCODE BIN
    # img = fluo_hdus[correspondings[all_timestamps[11]][1]]
    # # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.threshold(img, 2000, 2500, cv2.THRESH_BINARY)[1]
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    #
    # mixed_img = cv2.addWeighted(color, 0.4, cv2.cvtColor(fluo, cv2.COLOR_BGR2RGB).astype(np.uint8), 0.1, 0)
    # cv2.imshow('image', mixed_img)
    # cv2.waitKey(0)
    #
    # cv2.imshow('image', fluo_images[0])
    # cv2.waitKey(0)
