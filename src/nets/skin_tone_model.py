# Vendored from behra527/Skin-Tone-Classification-model: MobileNetV2 backbone (the repo's
# filename is mobilenetv2_skin_tone.h5, despite its own README describing VGG16) + a small
# classification head, reverse-engineered from that repo's saved-weight shapes since its
# model_config JSON does not deserialize on any Keras version tried (see README's Known
# Issues section) -- load_weights() against this matching architecture is the only path that
# gets past the file's own broken config, but a real weight-order mismatch inside the nested
# MobileNetV2 submodel means it STILL doesn't load cleanly as of this writing. No working
# pretrained weights currently ship for this feature -- see README.
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Sequential, Model

SKIN_TONE_INPUT_SHAPE = (90, 120, 3)  # (height, width, channels) -- this model's own idiosyncratic input size


def build_skin_tone_model(weights_path: str) -> Model:
    base = MobileNetV2(weights=None, include_top=False, input_shape=SKIN_TONE_INPUT_SHAPE)
    model = Sequential([
        base,
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(256, activation="relu"),
        Dense(3, activation="softmax"),
    ])
    model.build((None,) + SKIN_TONE_INPUT_SHAPE)
    model.load_weights(weights_path)
    return model
