from tensorflow import keras
from tcn import TCN, tcn_full_summary
import keras_tuner as kt
from keras_tuner import HyperModel
from utils.losses import MyWeightedBinaryCrossentropy

class TCNHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes, weights=None):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.weights = weights
    
    def build(self, hp):
        # Define the layers
        inputs = keras.layers.Input(shape=(None, self.input_shape[-1]), name="inputs")
        tcn = TCN(
            nb_filters=2**hp.Int("nb_filters", min_value=3, max_value=7, step=1),
            kernel_size=hp.Int("kernel_size", min_value=3, max_value=7, step=2),
            padding=hp.Choice("padding", ["causal", "same"], ordered=False, default="same"),
            dilations=[2**i for i in range(hp.Int("dilations", min_value=2, max_value=6, step=1))],
            return_sequences=True,
            name="tcn_layer"
        )(inputs)
        outputs = []
        for i in range(self.num_classes):
            outputs.append(keras.layers.Dense(units=1, activation="sigmoid", name=f"outputs_{i+1}")(tcn))
        
        # Instantiate the model
        model = keras.models.Model(inputs=inputs, outputs=outputs, name="tcn_model")

        # Define losses and metrics for each output
        losses, metrics = {}, {}
        for i in range(self.num_classes):
            if (self.weights is None) or (len(self.weights)==0):
                losses[f"outputs_{i+1}"] = MyWeightedBinaryCrossentropy(weights=0.01)
            else:
                losses[f"outputs_{i+1}"] = MyWeightedBinaryCrossentropy(weights=self.weights[f"outputs_{i+1}"])
            metrics[f"outputs_{i+1}"] = keras.metrics.BinaryAccuracy()
        
        # Compile the model
        model.compile(
            loss=losses,
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            metrics=metrics
        )
        return model

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