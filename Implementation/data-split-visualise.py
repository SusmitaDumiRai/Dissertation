import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from process_data import read_files

basename = "Thursday-15-02-2018_TrafficForML_CICFlowMeter"
data_loc = r"../Datasets/cleaned/{0}.csv".format(basename)
print(data_loc)
original_dataset = pd.read_csv(data_loc)
#original_dataset = read_files(data_loc, clean_data=False)

pd.plotting.register_matplotlib_converters()  # todo convert this to a function
original_dataset['Timestamp'] = pd.to_datetime(original_dataset['Timestamp'], format="%d/%m/%Y %H:%M:%S")
original_dataset = original_dataset.sort_values(['Timestamp'], ascending=[True]).reset_index(drop=True)

le_make = LabelEncoder()

original_dataset["Label-OHC"] = le_make.fit_transform(original_dataset["Label"])
#obj_df[["make", "make_code"]].head(11)

le_name_mapping = dict(zip(le_make.classes_, le_make.transform(le_make.classes_)))
print(le_name_mapping)
original_dataset.plot(x='Timestamp', y='Label-OHC')
plt.savefig(r"out/{0}/label-split".format(basename))
