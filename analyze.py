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


def make_anomalies(input_outputs, numsensors, normalizer=None):
    anomalies = []
    for element in input_outputs:
        if normalizer:
            anomalies.append(
                np.linalg.norm(
                    normalizer.inverse_transform(element[:numsensors].reshape(1, -1))
                    - normalizer.inverse_transform(element[numsensors:].reshape(1, -1))
                )
            )
        else:
            anomalies.append(
                np.linalg.norm(element[:numsensors] - element[numsensors:])
            )
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
                  batch_size: 100
        """
    )
    pipe = serializer.pipeline_from_definition(config)

    print("Fit model to first part of data")
    train_until = int(len(resampled_dataframe) / 4)
    model = pipe.fit(resampled_dataframe.iloc[:train_until])

    print("Run data through model for prediction")

    predict_from = 0
    res = model.transform(resampled_dataframe.iloc[predict_from:])
    anomalies = make_anomalies(res, 4)
    anomalies = pd.DataFrame(anomalies, index=resampled_dataframe.index[predict_from:])
    return anomalies


resamples_for_model = ["1S", "1T"]
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
        print(f"Build model for data resampled with {interval} interval")
        resampled = read_and_resample("2nd_test.hdf", interval)
        anomalies = build_model_return_anomalies(resampled)
        axarr[plotnum].plot(
            anomalies, label=interval + "-" + aggregation_method + "-model"
        )
        axarr[plotnum].legend(loc="upper left")
        plotnum += 1

plt.show()
