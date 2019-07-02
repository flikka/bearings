import itertools
import pandas as pd
import numpy as np
import yaml

from azureml.core import Workspace, Experiment
import mlflow

from sklearn.metrics import r2_score, explained_variance_score
from gordo_components import serializer

PLOTTING = False
if PLOTTING:
    import matplotlib.pyplot as plt


def read_and_resample(filename, resolution, aggregation_method="mean"):
    raw_frame = pd.read_hdf(filename)
    if aggregation_method == "max":
        return raw_frame.resample(resolution).max()
    elif aggregation_method == "mean":
        return raw_frame.resample(resolution).mean()


def make_anomalies(input_frame, outputs):
    anomalies = []
    for i in range(0, len(input_frame)):
        input_row = input_frame.iloc[i, :]
        output_row = outputs[i]
        anomalies.append(np.linalg.norm(input_row - output_row))

    return anomalies


def build_model(resampled_dataframe, epochs=5, batch_size=10):
    config = yaml.load(
        f"""
        sklearn.pipeline.Pipeline:
            steps:
              - sklearn.preprocessing.data.MinMaxScaler
              - gordo_components.model.models.KerasAutoEncoder:
                  kind: feedforward_hourglass
                  epochs: {epochs}
                  batch_size: {batch_size}


        """
    )
    pipe = serializer.pipeline_from_definition(config)

    print("Fit model to first part of data")
    train_until = int(len(resampled_dataframe) / 2)
    model = pipe.fit(resampled_dataframe.iloc[:train_until])

    print("Run data through model for prediction")
    predicted_data = model.predict(resampled_dataframe)
    anomalies = make_anomalies(resampled_dataframe, predicted_data)
    anomalies = pd.DataFrame(anomalies, index=resampled_dataframe.index)
    anomalies_mean_training = anomalies.iloc[:train_until].mean()[0]

    return (anomalies, anomalies_mean_training, predicted_data, train_until)


def calc_scores(resampled_dataframe, predicted_values, training_index_end):
    r2_train = r2_score(
        resampled_dataframe.values[:training_index_end],
        predicted_values[:training_index_end],
        multioutput="uniform_average",
    )
    expl_var_train = explained_variance_score(
        resampled_dataframe.values[:training_index_end],
        predicted_values[:training_index_end],
        multioutput="uniform_average",
    )

    r2_test = r2_score(
        resampled_dataframe.values[training_index_end:],
        predicted_values[training_index_end:],
        multioutput="uniform_average",
    )
    expl_var_test = explained_variance_score(
        resampled_dataframe.values[training_index_end:],
        predicted_values[training_index_end:],
        multioutput="uniform_average",
    )

    return (r2_train, expl_var_train, r2_test, expl_var_test)


def analyse_with_gordo():
    ws = Workspace.from_config()  # Azure ML
    # Get an experiment object from Azure Machine Learning
    experiment_name = "dummy_test"
    experiment = Experiment(workspace=ws, name=experiment_name)  # Azure ML
    mlflow.set_experiment(experiment_name)  # MLFlow

    resamples_for_model = ["1T", "1H"]
    aggregation_methods = ["max", "mean"]
    batch_sizes = [1, 10, 100]
    epochs = [1, 10]
    number_of_permutations = len(
        list(
            itertools.product(
                aggregation_methods, resamples_for_model, batch_sizes, epochs
            )
        )
    )

    resampled_original_data = read_and_resample("2nd_test.hdf", "1S")

    if PLOTTING:
        plotnum = 0
        f, axarr = plt.subplots(number_of_permutations + 1, sharex=True)
        axarr[plotnum].plot(
            resampled_original_data, linewidth=1, label="sensor_data_1S_mean"
        )
        axarr[plotnum].legend(loc="upper left")
        plotnum += 1

    for aggregation_method, interval, batch_size, epoch in itertools.product(
        aggregation_methods, resamples_for_model, batch_sizes, epochs
    ):
        run = experiment.start_logging()
        with mlflow.start_run():
            mlflow.log_param("interval", interval)  # MLFlow
            mlflow.log_param("aggregation_method", aggregation_method)  # MLFlow
            mlflow.log_param("batch_size", batch_size)  # MLFlow
            mlflow.log_param("epochs", epoch)  # MLFlow

            run.log("interval", interval)  # Azure ML
            run.log("aggregation_method", aggregation_method)  # Azure ML
            run.log("batch_size", batch_size)  # Azure ML
            run.log("epochs", epoch)  # Azure ML

            print(
                f"Build model for data resampled with {interval} resolution,  method {aggregation_method}, batch size {batch_size} and number of epochs {epoch}"
            )
            resampled = read_and_resample(
                "2nd_test.hdf", interval, aggregation_method=aggregation_method
            )
            anomalies, avg_train_anomaly, predicted_data, train_until_index = build_model(
                resampled, epoch, batch_size
            )

            r2_train, expl_train, r2_test, expl_test = calc_scores(
                resampled, predicted_data, train_until_index
            )
            run.log("r2_train", r2_train)  # Azure ML
            run.log("explained_variance_train", expl_train)  # Azure ML
            run.log("r2_test", r2_test)  # Azure ML
            run.log("explained_variance_test", expl_test)  # Azure ML

            mlflow.log_metric("r2_train", r2_train)  # MLFlow
            mlflow.log_metric("explained_variance_train", expl_train)  # MLFlow
            mlflow.log_metric("r2_test", r2_test)  # MLFlow
            mlflow.log_metric("explained_variance_test", expl_test)  # MLFlow

            anomalies = anomalies.rolling(
                resamples_for_model[-1]
            ).mean()  # Use the last of the experiment resamples as the anomaly resample
            if PLOTTING:
                axarr[plotnum].plot(
                    anomalies, label=interval + "-" + aggregation_method + "-model"
                )
                axarr[plotnum].axhline(avg_train_anomaly, color="r")
                axarr[plotnum].legend(loc="upper left")
                plotnum += 1

        run.complete()  # Azure ML

    if PLOTTING:
        plt.show()


if __name__ == "__main__":
    analyse_with_gordo()
