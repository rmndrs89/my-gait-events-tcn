import keras_tuner as kt
from .models import TCNHyperModel

def tune(train_data, train_targets, validation_data, weights=None, tuner_settings={}, search_settings={}):
    if len(tuner_settings)==0 or len(search_settings)==0:
        print(f"You need to specify tuner and/or search settings!")
        return
    print(f"Start hyperparameter tuning ...")
    # Define tuner    
    tuner = kt.RandomSearch(
        hypermodel=TCNHyperModel(
            input_shape=train_data.shape[1:],
            num_classes=len(train_targets),
            weights=weights
        ),
        objective="val_loss",
        max_trials=tuner_settings["max_trials"],
        executions_per_trial=tuner_settings["executions_per_trial"],
        overwrite=True,
        directory="/home/robbin/Desktop/tuning",
        project_name="bravo"
    )
    
    # Search hyperparameter space
    tuner.search(
        train_data,
        train_targets,
        epochs=search_settings["epochs"],
        validation_data=validation_data,
        shuffle=True,
        callbacks=search_settings["callbacks"],
        verbose=0
    )
    return tuner