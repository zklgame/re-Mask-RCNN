def _check_mode(mode):
    assert mode in ['training', 'inference']


def _check_shape(shape):
    '''
    Image size must be dividable by 2 multiple times
    :param shape: (h, w)
    :return: None
    '''
    h, w = shape
    if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
        raise Exception("Image size must be dividable by 2 at least 6 times "
                        "to avoid fractions when downscaling and upscaling."
                        "For example, use 256, 320, 384, 448, 512, ... etc. ")


def _check_resnet(resnet):
    assert resnet in ['resnet50', 'resnet101']