import os

import numpy as np
import cv2

from . import utils


class Transformer:
    def __init__(self, paths_config):
        self._DIR = paths_config['transformer_dir']
        os.makedirs(self._DIR, exist_ok=True)

    def transform(self, image_path: str) -> str:
        image = utils.load_image(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return self._save_image(image=image, initial_image_path=image_path)

    def _save_image(self, image: np.ndarray, initial_image_path: str) -> str:
        path = os.path.join(self._DIR, os.path.basename(initial_image_path))
        i = 1
        while os.path.exists(path):
            path = os.path.join(self._DIR, '_' * i + os.path.basename(initial_image_path))
            i += 1
        return utils.save_image(path, image)
