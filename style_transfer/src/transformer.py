import numpy as np
import cv2
from PIL import Image

from CIHP_master.PGN_single import PGN
from CIHP_master.utils_ import preds2coloredseg

from . import config
from .style_transfer import StyleTransferZoo


class SegmentationModel:
    def __init__(self):
        self._model = PGN()
        self._model.build_model(n_class=20, path_model_trained='./checkpoint/CIHP_pgn', tta=[0.75, 0.5],
                                img_size=(512, 512), need_edges=False)
        self._threshold = 0.7

    def predict(self, image: np.ndarray) -> (np.ndarray, np.ndarray):
        image = Image.fromarray(image, 'RGB')
        scores = self._model.predict(image)  # W x H x 20

        probs = np.ascontiguousarray(self._softmax(scores, axis=2).transpose(2, 0, 1))

        # Merge Shoes
        probs[-2] += probs[-1]
        probs = probs[:-1]

        assert len(probs) == len(config.SegmentationClassNames.ALL), len(probs)

        classes_ = np.argmax(probs, 0).astype(np.uint8)

        confidences = []
        for i in range(len(config.SegmentationClassNames.ALL)):
            nc = np.sum(classes_ == i)
            conf = 0 if nc == 0 else probs[i][classes_ == i].sum() / nc

            confidences.append(conf)
        confidences = np.array(confidences)

        labeled_image = np.array(preds2coloredseg(probs, image, out_format='gray'),
                                 dtype=np.uint8)
        return labeled_image, confidences

    @staticmethod
    def _softmax(x, axis=0):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x, axis)[..., None])
        return e_x / e_x.sum(axis=axis)[..., None]


class StyleTransferWithSegmentationModel:
    def __init__(self):
        self._segmentation_model = SegmentationModel()
        self._style_transformer = StyleTransferZoo()

    def alter_background(self, image: np.ndarray) -> (np.ndarray, np.ndarray):
        return self._transform_image(image=image,
                                     areas=config.SegmentationClassNames.BACKGROUND,
                                     style=config.Styles.HELL)

    def alter_cloths(self, image: np.ndarray) -> (np.ndarray, np.ndarray):
        return self._transform_image(image=image,
                                     areas=config.SegmentationClassNames.CLOTHS,
                                     style=config.Styles.ART)

    def _transform_image(self, image, areas, style):
        segmentation_mask, _ = self._segmentation_model.predict(image)
        transformed_image = self._style_transformer.transform(image=image,
                                                              style=style)
        return self._morph_transforms(initial_image=image,
                                      transformed_image=transformed_image,
                                      segmentation_mask=segmentation_mask,
                                      areas=areas)

    @staticmethod
    def _morph_transforms(initial_image, transformed_image,
                          segmentation_mask, areas):
        h, w = initial_image.shape[:2]
        transformed_image = cv2.resize(transformed_image, (w, h))
        assert transformed_image.shape == initial_image.shape, \
            '{} vs {}'.format(transformed_image.shape, initial_image.shape)

        result = initial_image.copy()
        affected_mask = np.zeros(shape=result.shape[:2], dtype=bool)
        for index, class_name in config.SegmentationClassNames.ALL:
            if class_name not in areas:
                continue
            this_class_mask = segmentation_mask == index
            result[this_class_mask] = transformed_image[this_class_mask]
            affected_mask[this_class_mask] = True
        return result, affected_mask