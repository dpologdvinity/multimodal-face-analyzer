# Vendored from serengil/deepface (MIT license): deepface/models/facial_recognition/VGGFace.py
# (embedding head -- load full VGGFace weights, then cut before the 2622-class softmax).
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten

from .deepface_common import vggface_base_model


def build_recognition_model(weights_path: str) -> Model:
    base = vggface_base_model()
    base.load_weights(weights_path)
    embedding = Flatten()(base.layers[-5].output)  # second 4096-d conv layer, pre-classification
    return Model(inputs=base.inputs[0], outputs=embedding)
