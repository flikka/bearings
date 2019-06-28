import pandas as pd
import numpy as np
import yaml

import matplotlib.pyplot as plt

from gordo_components import serializer


def read_and_resample(filename, resolution, aggregation_method="mean"):
    raw_frame = pd.read_hdf(filename)
    if aggregation_method == "max":
        return raw_frame.resample(resolution).max()
    elif aggregation_method == "mean":
        return raw_frame.resample(resolution).mean()


def make_anomalies(input_frame, outputs):
    anomalies = []
    for i in range(0, len(input_frame)):
        input_row = input_frame.iloc[i,:]
        output_row = outputs[i]
        anomalies.append(np.linalg.norm(input_row - output_row))

    return anomalies


def build_model_return_anomalies(resampled_dataframe):
    config = yaml.load(
        """
        sklearn.pipeline.Pipeline:
            steps:
              - sklearn.preprocessing.data.MinMaxScaler
              - gordo_components.model.models.KerasAutoEncoder:
                  kind: feedforward_hourglass
                  epochs: 3
                  batch_size: 10


        """
    )
    pipe = serializer.pipeline_from_definition(config)

    print("Fit model to first part of data")
    train_until = int(len(resampled_dataframe) / 5)
    model = pipe.fit(resampled_dataframe.iloc[:train_until])

    print("Run data through model for prediction")
    res = model.predict(resampled_dataframe)
    anomalies = make_anomalies(resampled_dataframe, res)
    anomalies = pd.DataFrame(anomalies, index=resampled_dataframe.index)
    anomalies_mean_training = anomalies.iloc[:train_until].mean()[0]
    return anomalies, anomalies_mean_training

def analyse_with_gordo():
    resamples_for_model = ["1T", "1H"]
    aggregation_methods = ["max", "mean"]
    plotnum = 0
    f, axarr = plt.subplots(
        len(aggregation_methods) * len(resamples_for_model) + 1, sharex=True
    )
    resampled_original_data = read_and_resample("2nd_test.hdf", "1S")
    axarr[plotnum].plot(resampled_original_data, linewidth=1, label="sensor_data_1S_mean")
    axarr[plotnum].legend(loc="upper left")
    plotnum += 1
    for aggregation_method in aggregation_methods:
        for interval in resamples_for_model:
            print(f"Build model for data resampled with {interval} interval and method {aggregation_method}")
            resampled = read_and_resample("2nd_test.hdf", interval, aggregation_method=aggregation_method)
            anomalies, avg_train_anomaly = build_model_return_anomalies(resampled)
            anomalies = anomalies.rolling(resamples_for_model[-1]).mean() # Use the last of the experiment resamples as the anomaly resample
            axarr[plotnum].plot(
                anomalies, label=interval + "-" + aggregation_method + "-model"
            )
            axarr[plotnum].axhline(avg_train_anomaly, color='r')
            axarr[plotnum].legend(loc="upper left")
            plotnum += 1

    plt.show()


analyse_with_gordo()
