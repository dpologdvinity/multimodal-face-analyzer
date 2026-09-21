# Vendored from serengil/deepface (MIT license): deepface/models/demography/Gender.py (gender head).
# Weights loaded separately from models/deepface_gender.h5 (gender_model_weights.h5).
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Convolution2D, Flatten, Activation

from .deepface_common import vggface_base_model


def build_gender_model(weights_path: str) -> Model:
    """Build the VGGFace-backbone gender classification head and load DeepFace's gender weights."""
    base = vggface_base_model()

    classes = 2
    output = Convolution2D(classes, (1, 1), name="predictions")(base.layers[-4].output)
    output = Flatten()(output)
    output = Activation("softmax")(output)

    gender_model = Model(inputs=base.inputs[0], outputs=output)
    gender_model.load_weights(weights_path)
    return gender_model
