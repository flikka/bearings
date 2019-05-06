import os
from datetime import datetime
from datetime import timedelta
import pandas as pd

def raw_to_pandas(directory):
    numlines_per_file = 20480
    delta = timedelta(minutes=10) / numlines_per_file
    dfs = []
    for filename in os.listdir(directory):
        filename_parts_ints = [int(element) for element in filename.split(".")]
        file_as_date = datetime(*filename_parts_ints)
        print(file_as_date)
        
        time_index = [file_as_date + delta*i for i in range(0, numlines_per_file)]

        current = pd.read_csv(os.path.join(directory, filename), sep="\t", header=None)
        current.index=time_index
        dfs.append(current)
        
        if len(current) != len(time_index):
            print("FEIL!" + filename)
            return
    
    return pd.concat(dfs)

if __name__=='__main__':
    df_with_dates = raw_to_pandas("2nd_test")
    df_with_dates.to_hdf("2nd_test.hdf", "raw")
    