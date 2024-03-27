from azureml.core.run import Run
import os
import argparse
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
import json

from ..training.steps.clean_data import clean_data
from ..training.steps.ingest_data import ingest_data
from ..training.steps.model_train import train_model
from .steps.evaluation import evaluation

if __name__ == '__main__':
    parser = argparse.ArgumentParser("train")
    parser.add_argument(
        "--config_suffix", type=str, help="Datetime suffix for json config files"
    )
    parser.add_argument(
        "--json_config",
        type=str,
        help="Directory to write all the intermediate json configs",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the Model",
        default="sklearn_regression_model.pkl",
    )

    args = parser.parse_args()

    print("Argument 1: %s" % args.config_suffix)
    print("Argument 2: %s" % args.json_config)

    model_name = args.model_name

    if not (args.json_config is None):
        os.makedirs(args.json_config, exist_ok=True)
        print("%s created" % args.json_config)

    run = Run.get_context()
    exp = run.experiment
    ws = run.experiment.workspace

    X, y = load_diabetes(return_X_y=True)
    columns = ["age", "gender", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)
    data = {"train": {"X": X_train, "y": y_train},
            "test": {"X": X_test, "y": y_test}}

    print("Running train.py")

    # Ingest data
    data = ingest_data("../data/diabetes2.csv")

    # Clean data
    X_train, X_test, y_train, y_test = clean_data(data)

    # Train the model

    # Get the model configuration from the config file
    with open("config.json") as f:
        config = json.load(f)

    # Train the model
    model = train_model(X_train, X_test, y_train, y_test, config)

    # Evaluate the model
    mse, r2, rmse = evaluation(model, X_test, y_test)
    run.log("mse", mse)
    run.log("r2", r2)
    run.log("rmse", rmse)

    # Save model as part of the run history

    # model_name = "."

    with open(model_name, "wb") as file:
        joblib.dump(value=model, filename=model_name)

    # upload the model file explicitly into artifacts
    run.upload_file(name="./outputs/" + model_name, path_or_stream=model_name)
    print("Uploaded the model {} to experiment {}".format(
        model_name, run.experiment.name))
    dirpath = os.getcwd()
    print(dirpath)
    print("Following files are uploaded ")
    print(run.get_file_names())

    run_id = {}
    run_id["run_id"] = run.id
    run_id["experiment_name"] = run.experiment.name
    filename = "run_id_{}.json".format(args.config_suffix)
    output_path = os.path.join(args.json_config, filename)
    with open(output_path, "w") as outfile:
        json.dump(run_id, outfile)

    run.complete()
