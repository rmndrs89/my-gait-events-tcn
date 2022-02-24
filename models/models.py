from .losses import MyWeightedBinaryCrossentropy
import tensorflow as tf
from tensorflow import keras
from tcn import TCN, tcn_full_summary

def build_model(input_dim, class_names, loss_fn=MyWeightedBinaryCrossentropy(), **kwargs):
    # Define layers
    inputs = keras.layers.Input(shape=(None, input_dim), name="inputs")
    tcn = TCN(
        return_sequences=True,
        use_batch_norm=True,
        name="tcn_layer",
        **kwargs
    )(inputs)
    outputs = []
    for class_name in class_names:
        outputs.append(
            keras.layers.Dense(units=1, activation="sigmoid", name=class_name)(tcn)
        )
    
    # Instantiate the model
    model = keras.models.Model(inputs=inputs, outputs=outputs, name="tcn_model")
    
    # For each output, define the loss function
    losses = {}
    for class_name in class_names:
        losses[class_name] = loss_fn
    
    # Compile the model
    model.compile(
        loss=losses,
        optimizer=keras.optimizers.Adam(learning_rate=0.001)
    )
    return model