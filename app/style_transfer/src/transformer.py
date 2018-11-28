import random

import numpy as np
import cv2
from albumentations.augmentations.functional import shift_hsv

from app.style_transfer import config


class StyleTransferWithSegmentationModel:
    def __init__(self):
        self._segmentation_model = SegmentationModel()
        # self._style_transformer = StyleTransferZoo()  FIXME

    def alter_background(self, image_path: str) -> (np.ndarray, np.ndarray):
        return self._transform_image(image_path=image_path,
                                     areas=config.SegmentationClassNames.BACKGROUND,
                                     style=config.Styles.HELL, cloths=False)

    def alter_cloths(self, image_path: str) -> (np.ndarray, np.ndarray):
        return self._transform_image(image_path=image_path,
                                     areas=config.SegmentationClassNames.CLOTHS,
                                     style=config.Styles.ART, cloths=True)

    def _transform_image(self, image_path, areas, style, cloths):
        segmentation_mask, _ = self._segmentation_model.predict(image_path)
        # transformed_image = self._style_transformer.transform(image=image, style=style)  FIXME
        image = cv2.imread(image_path)
        assert image is not None, image_path
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # return image, None
        if not cloths:
            transformed_image = self._motion_blur(image, ksize=45)
        else:
            transformed_image = shift_hsv(image, 100, 50, 50)
        return self._morph_transforms(initial_image=image,
                                      transformed_image=transformed_image,
                                      segmentation_mask=segmentation_mask,
                                      areas=areas)



    @staticmethod
    def _motion_blur(img, ksize):
        assert ksize > 2
        kernel = np.zeros((ksize, ksize), dtype=np.uint8)
        xs, xe = random.randint(0, ksize - 1), random.randint(0, ksize - 1)
        if xs == xe:
            ys, ye = random.sample(range(ksize), 2)
        else:
            ys, ye = random.randint(0, ksize - 1), random.randint(0, ksize - 1)
        cv2.line(kernel, (xs, ys), (xe, ye), 1, thickness=1)
        return cv2.filter2D(img, -1, kernel / np.sum(kernel))
