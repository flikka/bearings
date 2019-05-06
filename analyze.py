import pandas as pd
import numpy as np
import yaml

import matplotlib.pyplot as plt

from gordo_components import serializer

def read_and_resample(filename, resolution):
    raw_frame = pd.read_hdf(filename)
    return raw_frame.resample(resolution).mean()


def make_anomalies(input_outputs, numsensors):
    anomalies = []
    for element in input_outputs:
        anomalies.append(np.linalg.norm(element[:numsensors] - element[numsensors:]))
    return anomalies

resampled = read_and_resample("2nd_test.hdf", "1T")

config = yaml.load(
    """ 
    sklearn.pipeline.Pipeline:
        steps:
          - sklearn.preprocessing.data.MinMaxScaler
          - gordo_components.model.models.KerasAutoEncoder:
              kind: feedforward_hourglass
              epochs: 10
    """
)
pipe = serializer.pipeline_from_definition(config)

model = pipe.fit(resampled.ix[:int(len(resampled) / 2)])

res = model.transform(resampled)
anomalies = make_anomalies(res, 4)
anomalies = pd.DataFrame(anomalies, index=resampled.index)

plt.figure()
plt.plot(resampled)

plt.plot(anomalies)

