import os
import enum

import numpy as np
import cv2

from . import utils
from .face_swapper import FaceSwapper
from .segmentation_style_transfer import SegmentationStyleTransfer


@enum.unique
class AnonymizerActions(enum.Enum):
    PANTS_TO_JEANS = enum.auto()
    PANTS_TO_SKIN = enum.auto()
    COAT_TO_JEANS = enum.auto()
    COAT_TO_SKIN = enum.auto()
    COLORIZE_DRESS = enum.auto()
    COLORIZE_SHOES = enum.auto()

    ALL_IN = enum.auto()

    BACKGROUND_WINTER = enum.auto()
    BACKGROUND_SUMMER = enum.auto()
    BACKGROUND_INFERNO = enum.auto()
    BACKGROUND_UNDEAD = enum.auto()


STYLE_TRANSFER_PARAMS = {
    AnonymizerActions.PANTS_TO_JEANS: {
        'Pants': 'jeans'
    },
    AnonymizerActions.PANTS_TO_SKIN: {
        'Pants': 'crocodile',
        'Dress': 'la_muse'
    },

    AnonymizerActions.COAT_TO_JEANS: {
        'Coat': 'jeans',
        'Skirt': 'jeans',
        'Upper clothes': 'wave'

    },
    AnonymizerActions.COAT_TO_SKIN: {
        'Coat': 'crocodile',
        'Shoes': 'rain_princess',
        'Skirt': 'crocodile',
        'Upper clothes': 'la_muse'
    },

    AnonymizerActions.COLORIZE_DRESS: {
        'Upper clothes': 'udnie',
        'Dress': 'wave',
        'Shoes': 'scream',
        'Coat': 'la_muse'
        # 'Background': 'summer'
    },
    AnonymizerActions.COLORIZE_SHOES: {
        'Shoes': 'la_muse',
        'Pants': 'jeans',
        'Dress': 'udnie',
        'Skirt': 'rain_princess',
        'Upper clothes': 'scream'
    },

    AnonymizerActions.ALL_IN: {
        'Pants': 'jeans',
        'Coat': 'crocodile',
        'Upper clothes': 'udnie',
        'Shoes': 'la_muse',
        'Background': 'winter',
        'Dress': 'rain_princess'
    },

    AnonymizerActions.BACKGROUND_SUMMER: {
        'Pants': 'scream',
        'Skirt': 'la_muse',
        'Coat': 'udnie',
        #   'Background': 'rain_princess'
    },
    #
    # AnonymizerActions.BACKGROUND_WINTER: {
    #     'Background': 'inferno'
    # },

    # AnonymizerActions.BACKGROUND_SUMMER: {
    #     'Background': 'summer'
    # },

    #,

    AnonymizerActions.BACKGROUND_UNDEAD: {
        'Background': 'undead',
        'Shoes': 'scream'
    }
}

PAD_SIZE = 32


class Anonymizer:
    def __init__(self, paths_config):
        self._DIR = paths_config['anonymizer_dir']
        os.makedirs(self._DIR, exist_ok=True)

        self._face_swapper = FaceSwapper(paths_config)
        self._style_transfer = SegmentationStyleTransfer(paths_config)

    def anonymize(self, image_path: str, action) -> str:
        image = utils.load_image(image_path)
        image = cv2.copyMakeBorder(image, PAD_SIZE, PAD_SIZE, PAD_SIZE, PAD_SIZE, cv2.BORDER_REPLICATE)
        modified_image = image.copy()

        # if actions[AnonymizerActions.FACE_SWAP]:
        #     modified_image = self._face_swap(image=image)

        style_transfer_params = STYLE_TRANSFER_PARAMS[action]
        # if actions[AnonymizerActions.CLOTHS_STYLE_TRANSFER] or actions[AnonymizerActions.BACKGROUND_STYLE_TRANSFER]:
        modified_image = self._segmentation_style_transfer(
            init_path=image_path,
            image=modified_image,
            params=style_transfer_params
        )
        if action == AnonymizerActions.ALL_IN:
            modified_image = np.vstack([self._remove_padding(image), self._remove_padding(modified_image)])
        else:
            modified_image = self._remove_padding(modified_image)
        return self._save_image(image=modified_image, initial_image_path=image_path)

    def _remove_padding(self, image):
        image = image[PAD_SIZE: image.shape[0] - PAD_SIZE, PAD_SIZE: image.shape[1] - PAD_SIZE]
        return image

    def compose_bottom(self, image_path: str) -> str:
        image = utils.load_image(image_path)
        h, w = image.shape[:2]
        # image = cv2.copyMakeBorder(image, PAD_SIZE, PAD_SIZE, PAD_SIZE, PAD_SIZE, cv2.BORDER_REPLICATE)

        num_h = 3
        num_w = 2
        result = np.zeros((num_h * h, num_w * w, 3),
                          dtype=np.uint8)

        result[:h, :w] = utils.load_image(self.anonymize(image_path, AnonymizerActions.COAT_TO_JEANS))
        result[:h, w:2 * w] = utils.load_image(self.anonymize(image_path, AnonymizerActions.COLORIZE_SHOES))

        result[h:2 * h, :w] = utils.load_image(self.anonymize(image_path, AnonymizerActions.PANTS_TO_SKIN))
        result[h:2 * h, w:2 * w] = utils.load_image(self.anonymize(image_path, AnonymizerActions.COAT_TO_SKIN))

        result[2 * h:3 * h, :w] = self._face_swap(utils.load_image(self.anonymize(image_path, AnonymizerActions.BACKGROUND_SUMMER)))
        result[2 * h:3 * h, w:2 * w] = self._face_swap(utils.load_image(self.anonymize(image_path, AnonymizerActions.COLORIZE_DRESS)))

        # result[3 * h:4 * h, :w] = self._face_swap(
        #     utils.load_image(self.anonymize(image_path, AnonymizerActions.BACKGROUND_INFERNO)))
        # result[3 * h:4 * h, w:2 * w] = self._face_swap(
        #     utils.load_image(self.anonymize(image_path, AnonymizerActions.BACKGROUND_WINTER)))

        result = cv2.resize(result, (w, int(1.5 * h)))

        return self._save_image(image=result, initial_image_path=image_path)

    def _save_image(self, image, initial_image_path):
        path = os.path.join(self._DIR, os.path.basename(initial_image_path))
        i = 1
        while os.path.exists(path):
            path = os.path.join(self._DIR, '_' * i + os.path.basename(initial_image_path))
            i += 1
        return utils.save_image(path, image)

    def _face_swap(self, image):
        return self._face_swapper.transform(image)

    def _segmentation_style_transfer(self, init_path, image, params):
        return self._style_transfer.transform(init_path=init_path, image=image, params=params)
