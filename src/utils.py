import argparse
import json


class Constants:
    # Label Mapping
    LANG_LABEL = ["fr", "en"]
    COL_LABEL_NAME = "labelling_name_"
    COL_LABEL_ID = "labelling_id"

    # Classification
    POSITIVE = (1, "contain_tomato")
    NEGATIVE = (0, "not_contain")

    # Dataset
    DEFAULT_PARAM_LOADER = {
        "batch_size": 32,
        "validation_split": 0.2,
        "shuffle_dataset": true,
        "random_seed": 42
    }

    DEFAULT_PARAM_CLASSIFIER = {
        "threshold": 0.5,
        "learning_rate": 0.001,
        "epochs": 40,
        "loss_weight": [1, 6]
    }


def init_parser_train():
    """
    Function used to initialize parameters of script.

    - `folder_imgs`: path folder with image dataset
    - `img_annotations`: path file for `the annotations.json`
    - `label_mapping`: path file for the `label_mapping.csv`
    - `model_filepath`: path file to save the model checkpoint
    - `config`: path file for the `config.json` file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_imgs", help="Folder path with image dataset")
    parser.add_argument("img_annotations", help="File path for the annotations.json")
    parser.add_argument("label_mapping", help="File path for the label_mapping.csv")
    parser.add_argument("model_filepath", help="File path to save the model checkpoint")
    parser.add_argument("config", help="File path for the config.json file")
    return parser.parse_args()


def init_parser_pred():
    """
    Function used to initialize parameters of script pred.

    - `model_filepath`: filepath for the model checkpoint
    - `img_filepath`: filepath for the image we want to predict class
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("model_filepath", help="File path for the model")
    parser.add_argument(
        "img_filepath", help="File path for the image we want to predict class"
    )
    parser.add_argument("--plot_result", help="Boolean to plot or not the result")
    return parser.parse_args()


def parse_config_file(config_path):
    """
    Function to parse json config file.
    """
    with open(config_path) as json_f:
        config = json.load(json_f)
    return (config["param_loader"], config["param_classifier"], config["param_common"])
