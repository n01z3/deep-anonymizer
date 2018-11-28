import os
import enum

from . import utils
from .face_swapper import FaceSwapper
from .segmentation_style_transfer import SegmentationStyleTransfer


@enum.unique
class AnonymizerActions(enum.Enum):
    FACE_SWAP = enum.auto()
    CLOTHS_STYLE_TRANSFER = enum.auto()
    BACKGROUND_STYLE_TRANSFER = enum.auto()


class Anonymizer:
    def __init__(self, paths_config):
        self._DIR = paths_config['anonymizer_dir']
        os.makedirs(self._DIR, exist_ok=True)

        self._face_swapper = FaceSwapper(paths_config)
        self._style_transfer = SegmentationStyleTransfer(paths_config)

    def anonymize(self, image_path: str, actions: dict) -> str:
        image = utils.load_image(image_path)
        modified_image = image.copy()

        if actions[AnonymizerActions.FACE_SWAP]:
            modified_image = self._face_swap(image=image)

        if actions[AnonymizerActions.CLOTHS_STYLE_TRANSFER] or actions[AnonymizerActions.BACKGROUND_STYLE_TRANSFER]:
            modified_image = self._segmentation_style_transfer(
                image=modified_image,
                cloths=actions[AnonymizerActions.CLOTHS_STYLE_TRANSFER],
                background=actions[AnonymizerActions.BACKGROUND_STYLE_TRANSFER]
            )
        return self._save_image(image=modified_image, initial_image_path=image_path)

    def _save_image(self, image, initial_image_path):
        path = os.path.join(self._DIR, os.path.basename(initial_image_path))
        return utils.save_image(path, image)

    def _face_swap(self, image):
        return self._face_swapper.transform(image)

    def _segmentation_style_transfer(self, image, cloths, background):
        return self._style_transfer.transform(image=image, cloths=cloths, background=background)
