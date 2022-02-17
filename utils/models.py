from tensorflow import keras
from tcn import TCN, tcn_full_summary
import keras_tuner as kt

from utils.losses import MyWeightedBinaryCrossentropy

def get_base_model(input_shape, num_classes, weights=None, **kwargs):

    # Define the model architecture
    inputs = keras.layers.Input(shape=(None, input_shape[-1]), name="inputs")
    tcn = TCN(
        return_sequences=True, 
        name="tcn_layer"
    )(inputs)
    outputs = []
    for i in range(num_classes):
        outputs.append(keras.layers.Dense(units=1, activation="sigmoid", name=f"outputs_{i+1}")(tcn))

    # Instantiate the model
    tcn_model = keras.models.Model(inputs=inputs, outputs=outputs, name="tcn_model")

    # Define losses and metrics for each output
    losses, metrics = {}, {}
    for i in range(num_classes):
        if (weights is None) or (len(weights)==0):
            losses[f"outputs_{i+1}"] = MyWeightedBinaryCrossentropy(weights=0.01)
        else:
            losses[f"outputs_{i+1}"] = MyWeightedBinaryCrossentropy(weights=weights[f"outputs_{i+1}"])
        metrics[f"outputs_{i+1}"] = keras.metrics.BinaryAccuracy()

    # Compile the model
    tcn_model.compile(
        loss=losses,
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=metrics
    )
    return tcn_model