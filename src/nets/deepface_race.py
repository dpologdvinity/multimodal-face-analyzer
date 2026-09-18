# Vendored from serengil/deepface (MIT license): deepface/models/demography/Race.py (race head).
# Weights loaded separately from models/deepface_race.h5 (race_model_single_batch.h5).
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Convolution2D, Flatten, Activation

from .deepface_common import vggface_base_model


def build_race_model(weights_path: str) -> Model:
    """Build the VGGFace-backbone race classification head and load DeepFace's race weights."""
    base = vggface_base_model()

    classes = 6
    output = Convolution2D(classes, (1, 1), name="predictions")(base.layers[-4].output)
    output = Flatten()(output)
    output = Activation("softmax")(output)

    race_model = Model(inputs=base.inputs, outputs=output)
    race_model.load_weights(weights_path)
    return race_model
