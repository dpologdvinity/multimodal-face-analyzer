# Vendored from chandrikadeb7/Face-Mask-Detection (MIT license): train_mask_detector.py's
# model definition. Weights loaded separately from models/mask_detector.h5 (that repo's own
# mask_detector.model, an HDF5 file saved via model.save(..., save_format="h5")).
# Loaded via load_weights() rather than a full load_model() -- the saved file's Keras-version
# metadata doesn't deserialize cleanly under this repo's newer Keras, but the weight arrays
# still apply correctly onto a freshly-built, architecture-identical model.
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model


def build_mask_model(weights_path: str) -> Model:
    base = MobileNetV2(weights=None, include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    head = base.output
    head = AveragePooling2D(pool_size=(7, 7))(head)
    head = Flatten(name="flatten")(head)
    head = Dense(128, activation="relu")(head)
    head = Dropout(0.5)(head)
    head = Dense(2, activation="softmax")(head)
    model = Model(inputs=base.input, outputs=head)
    model.load_weights(weights_path)
    return model
