from datasets import keepcontrol

def main():
    # Load dataset
    ds_train, ds_val, ds_test = keepcontrol.load_data(path=PATH,
                                                      filename=DEMOGRAPHICS_FILE,
                                                      tracked_points=TRACKED_POINTS,
                                                      classification_task=CLASSIFICATION_TASK,
                                                      win_len=WIN_LEN)
    return

if __name__ == "__main__":
    # Global variables
    PATH = "/mnt/neurogeriatrics_data/Keep Control/Data/lab dataset/rawdata"
    DEMOGRAPHICS_FILE = "/mnt/neurogeriatrics_data/Keep Control/Data/lab dataset/rawdata/participants.tsv"
    TRACKED_POINTS = ["left_ankle", "right_ankle"]
    CLASSIFICATION_TASK = "events"
    WIN_LEN = 400
    
    # Call main function
    main()