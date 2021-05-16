from typing import List, Dict, Tuple

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import cv2

import os


class FluoMetaData:
    def __init__(self, hdu: fits.PrimaryHDU):
        self.TRANSC0 = hdu.header.cards['TRANSC0'].value
        self.TRANSC1 = hdu.header.cards['TRANSC1'].value
        self.TRANSC2 = hdu.header.cards['TRANSC2'].value
        self.TRANSC3 = hdu.header.cards['TRANSC3'].value
        self.TRANSC4 = hdu.header.cards['TRANSC4'].value
        self.TRANSC5 = hdu.header.cards['TRANSC5'].value
        self.TRANSC6 = hdu.header.cards['TRANSC6'].value
        self.TRANSC7 = hdu.header.cards['TRANSC7'].value
        self.TRANSC8 = hdu.header.cards['TRANSC8'].value

    def get_warp_matrix(self) -> np.ndarray:
        return np.array([
            [self.TRANSC0, self.TRANSC1, self.TRANSC2],
            [self.TRANSC3, self.TRANSC4, self.TRANSC5],
            [self.TRANSC6, self.TRANSC7, self.TRANSC8]
        ])


def get_hdu_dictionary(hdu_list: fits.HDUList) -> Dict[int, np.ndarray]:
    ret = {}
    for idx in range(1, len(hdu_list)):
        timestamp_index = int(hdu_list[idx].header.cards["TIMESTMP"].value)
        color_data = hdu_list[idx].data
        ret[timestamp_index] = color_data
    return ret


def get_corresponding(colors: Dict[int, np.ndarray], fluos: Dict[int, np.ndarray]) -> (List[str], Dict[int, List]):
    all_timestamp = list({**colors, **fluos})
    latest_color = min(colors)
    latest_fluo = min(fluos)
    colors = sorted(colors)
    fluos = sorted(fluos)
    corresponding_images = {}
    for t in all_timestamp:
        latest_color = t if t in colors else latest_color
        latest_fluo = t if t in fluos else latest_fluo
        corresponding_images[t] = [latest_color, latest_fluo]
    return all_timestamp, corresponding_images


def resize_fluo_images(imgs: Dict[int, np.ndarray], shape: Tuple[int, int]):
    ret: Dict[int, np.ndarray] = {}
    for idx in imgs:
        ret[idx] = cv2.resize(imgs[idx], shape, interpolation=cv2.INTER_AREA)
    return ret


def remove_bayer_pattern(imgs: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    ret: Dict[int, np.ndarray] = {}
    for idx in imgs:
        ret[idx] = cv2.cvtColor(imgs[idx].astype(np.uint8), cv2.COLOR_BAYER_BG2BGR)
    return ret


def fluo_warp_perspective(imgs: Dict[int, np.ndarray], register_params: FluoMetaData) -> Dict[int, np.ndarray]:
    ret: Dict[int, np.ndarray] = {}
    for idx in imgs:
        ret[idx] = cv2.warpPerspective(imgs[idx], register_params.get_warp_matrix(),
                                       (imgs[idx].shape[1], imgs[idx].shape[0]))
    return ret


def flip_images(imgs: Dict[int, np.ndarray], flip_value=0) -> Dict[int, np.ndarray]:
    ret: Dict[int, np.ndarray] = {}
    for idx in imgs:
        ret[idx] = cv2.flip(imgs[idx], flip_value)
    return ret


def fluo_to_color(imgs: Dict[int, np.ndarray]):
    ret: Dict[int, np.ndarray] = {}
    for idx in imgs:
        ret[idx] = cv2.cvtColor(imgs[idx], cv2.COLOR_BGR2RGB)
    return ret


if __name__ == '__main__':
    file_dir: str = os.path.dirname(os.path.realpath(__file__))
    color_dir: str = os.path.join(file_dir, 'Mice2_cetu2_131213_210250_color.fits')
    fluo_dir: str = os.path.join(file_dir, 'Mice2_cetu2_131213_210250_fluo.fits')
    # first element is meta data
    color_data_list: fits.HDUList = fits.open(color_dir)
    fluo_data_list: fits.HDUList = fits.open(fluo_dir)
    fluo_meta = FluoMetaData(fluo_data_list[0])
    color_hdus = get_hdu_dictionary(color_data_list)  # smallest=3701402 len=1061 shape=(1024,1392)
    fluo_hdus = get_hdu_dictionary(fluo_data_list)  # smallest=3701230 len=1406 shape=(256,256)
    # close the fits file
    color_data_list.close()
    fluo_data_list.close()

    # color_hdus = resize_fluo_images(color_hdus, (1024, 1392))

    # col = cv2.cvtColor(, cv2.COLOR_BAYER_BG2BGR)

    # TODO PROCESS IMAGES HERE
    #   scale fluo to color
    #   img = color_img.astype(np.uint8)
    #     bgr_img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)
    #   todo exctract registraiton params
    #   todo register fluo images
    #     fluo_img_registred = cv2.warpPerspective(fluo_img, warp_matrix, (col_width,col_height))
    #     col_width = shape.x ...
    # opencv warp_perspective

    all_timestamps, correspondings = get_corresponding(color_hdus, fluo_hdus)

    color_hdus = remove_bayer_pattern(color_hdus)
    color_hdus = fluo_to_color(color_hdus)
    fluo_hdus = resize_fluo_images(fluo_hdus,
                                   (color_hdus[all_timestamps[0]].shape[1],
                                    color_hdus[all_timestamps[0]].shape[0]))

    fluo_hdus = flip_images(fluo_hdus, 1)
    fluo_hdus = fluo_warp_perspective(fluo_hdus, fluo_meta)

    f, axarr = plt.subplots(1, 2)

    color_plot_image = axarr[0].imshow(color_hdus[correspondings[all_timestamps[0]][0]])
    axarr[0].title.set_text('ColorFit')
    axarr[0].set_xticks([])
    axarr[0].set_yticks([])

    fluo_plot_image = axarr[1].imshow(fluo_hdus[correspondings[all_timestamps[0]][1]], cmap='gray')
    axarr[1].title.set_text('FluoFit')
    axarr[1].set_xticks([])
    axarr[1].set_yticks([])

    slider_value = 0
    axslid = plt.axes([0.1, 0.05, 0.8, 0.075])

    slider = plt.Slider(axslid, 'Slider', 0,
                        len(all_timestamps), valstep=1, valinit=slider_value)


    def update(val):
        color_image = color_hdus[correspondings[all_timestamps[val]][0]]
        fluo_image = fluo_hdus[correspondings[all_timestamps[val]][1]]
        color_plot_image.set_data(color_image)
        fluo_plot_image.set_data(fluo_image)
        plt.draw()


    overlayed = []
    for t in range(len(all_timestamps)):
        color = color_hdus[correspondings[all_timestamps[t]][0]]
        fluo = cv2.cvtColor(fluo_hdus[correspondings[all_timestamps[t]][1]], cv2.COLOR_GRAY2RGB).astype(np.uint8)
        mixed_img = cv2.addWeighted(color, 0.4, fluo, 0.1, 0)
        overlayed.append(mixed_img)

    # writer = cv2.VideoWriter(os.path.join(file_dir, 'video.mp4'), cv2.VideoWriter_fourcc(*'XVID'), 20, (1392, 1024),
    #                          True)
    writer = cv2.VideoWriter(os.path.join(file_dir, 'video.mp4'), -1, 20.0, (1392, 1024))

    for i in overlayed:
        writer.write(i)

    slider.on_changed(update)

    plt.show()
    print('end')
