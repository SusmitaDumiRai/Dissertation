import glob
import pandas as pd
import sys

filenames = [i for i in glob.glob("{0}/*.csv".format(sys.argv[1]))]

print(filenames)
print("Number of files: {0}".format(len(filenames)))
combined_csv = pd.concat([pd.read_csv(f) for f in filenames])
combined_csv.to_csv("{0}/merged_csv.csv".format(sys.argv[1]), index=False)
