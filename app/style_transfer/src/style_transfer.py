import numpy as np

from app.style_transfer import config


class AbstractStyleTransferModel:
    def __init__(self, supported_styles):
        self._supported_styles = supported_styles

    def transform(self, image: np.ndarray, style: str) -> np.ndarray:
        if style not in self._supported_styles:
            raise ValueError('Unknown style "{}"'.format(style))
        return self._make_transform(image, style)

    def get_supported_styles(self):
        return self._supported_styles

    def _make_transform(self, image: np.ndarray, style: str) -> np.ndarray:
        raise NotImplementedError


class NvidiaStyleTransfer(AbstractStyleTransferModel):
    def __init__(self):
        super().__init__(config.SupportedStyles.NVIDIA)

    def _make_transform(self, image: np.ndarray, style: str):
        raise NotImplementedError  # TODO: to be implemented


class FastStyleTransfer(AbstractStyleTransferModel):
    def __init__(self):
        super().__init__(config.SupportedStyles.FAST_STYLE_TRANSFER)

    def _make_transform(self, image: np.ndarray, style: str):
        raise NotImplementedError  # TODO: to be implemented


class StyleTransferZoo:
    def __init__(self):
        self._zoo = self._build_zoo([NvidiaStyleTransfer(), FastStyleTransfer()])

    def transform(self, image: np.ndarray, style) -> np.ndarray:
        return self._zoo[style].transform(image)

    def _build_zoo(self, models):
        zoo = dict()
        for model in models:
            zoo = self._register_model(zoo, model)
        return zoo

    @staticmethod
    def _register_model(zoo: dict, model: AbstractStyleTransferModel):
        for style in model.get_supported_styles():
            zoo[style] = model
        return zoo
