import numpy as np
from PIL import Image

from .CIHP_master.PGN_single import PGN
from .CIHP_master.utils_ import preds2coloredseg
from . import config


class SegmentationModel:
    def __init__(self, weights_dir):
        self._model = PGN()
        self._model.build_model(n_class=20, path_model_trained=weights_dir,
                                tta=[0.75, 0.5],
                                img_size=(512, 512), need_edges=False)
        self._threshold = 0.7

    def predict(self, image_path: str) -> (np.ndarray, np.ndarray):
        # image = Image.fromarray(image, 'RGB')

        scores = self._model.predict(image_path)  # W x H x 20

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

        image = Image.open(image_path)
        labeled_image = np.array(preds2coloredseg(probs, image, out_format='gray'),
                                 dtype=np.uint8)
        return labeled_image, probs

    @staticmethod
    def _softmax(x, axis=0):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x, axis)[..., None])
        return e_x / e_x.sum(axis=axis)[..., None]
