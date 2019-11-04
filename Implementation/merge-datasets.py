import glob
import pandas as pd

filenames = [i for i in glob.glob("../Datasets/*.csv")]

# print(filenames)

combined_csv = pd.concat([pd.read_csv(f) for f in filenames])
combined_csv.to_csv("merged_csv.csv", index=False)
