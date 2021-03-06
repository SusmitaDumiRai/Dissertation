from os.path import basename
import glob
import pandas as pd

#filenames = [i for i in glob.glob(r"C:\Users\susi\Documents\aws-dataset\original\Friday-16-02-2018_TrafficForML_CICFlowMeter.csv")]

filenames = ["merged_csv.csv"]

# print(filenames)

for file in filenames:
  if file != 'merged-csv.csv':
    print("reading file: %s" % file)
    df = pd.read_csv(file)
    df = df[df.Label.str.contains('Label') == False]  # Removes csv rows that have headers repeating
    print("dropping nans")
    df = df.dropna(how='any')
    print("Any nans?: {0}".format(df.isnull().values.any()))
    print("Removing infinity")
    for col in list(df):
      # print("Number of infinities before: + {0}".format((df[col] == 'Infinity').sum()))

      df = df.drop(df[df[col] == 'Infinity'].index)
      # print("Number of infinities after: + {0}".format((df[col] == 'Infinity').sum()))

    df = df.reset_index(drop=True)
    print("writing file out")
    base = basename(file)
    print("Basename: {0}".format(base))
    df.to_csv(r"C:\Users\susi\Documents\aws-dataset/cleaned/{0}".format(base), index=False)
