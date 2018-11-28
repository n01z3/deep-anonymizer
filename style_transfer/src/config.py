import enum


class SegmentationClassNames:
    ALL = (
        'Background',
        'Hat',
        'Hair',
        'Glove',
        'Sunglasses',
        'Upper clothes',
        'Dress',
        'Coat',
        'Socks',
        'Pants',
        'Torso skin',
        'Scarf',
        'Skirt',
        'Face',
        'Left arm',
        'Right arm',
        'Left leg',
        'Right leg',
        'Shoes'
    )

    CLOTHS = (
        'Hat',
        'Glove',
        'Sunglasses',
        'Upper clothes',
        'Dress',
        'Coat',
        'Socks',
        'Pants',
        'Scarf',
        'Skirt',
        'Shoes'
    )

    BACKGROUND = (
        'Background'
    )


class Styles(enum.Enum):
    HELL = 'Hell'
    ART = 'Art'


class SupportedStyles(enum.Enum):
    NVIDIA = {Styles.HELL}
    FAST_STYLE_TRANSFER = {Styles.ART}
