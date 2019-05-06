import pandas as pd
import numpy as np
import yaml

import matplotlib.pyplot as plt

from gordo_components import serializer


def read_and_resample(filename, resolution):
    raw_frame = pd.read_hdf(filename)
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

    predict_from = train_until * 3
    res = model.transform(resampled_dataframe.iloc[predict_from:])

    anomalies = make_anomalies(res, 4, normalizer=pipe.steps[0][1])
    anomalies = pd.DataFrame(anomalies, index=resampled_dataframe.index[predict_from:])
    return anomalies


resamples_for_model = ["1S", "30S", "1T"]

resulting_resampling = "5T"
resampled_original_data = read_and_resample("2nd_test.hdf", resulting_resampling)
plt.figure()
plt.plot(resampled_original_data, linewidth=0.5)
for interval in resamples_for_model:
    print(f"Build model for data resampled with {interval} interval")
    resampled = read_and_resample("2nd_test.hdf", interval)
    anomalies = (
        build_model_return_anomalies(resampled).resample(resulting_resampling).mean()
    )

    plt.plot(anomalies, linewidth=2, linestyle="dashed", label=interval + "-model")

plt.legend(loc="upper left")
plt.show()
