import numpy as np
import pandas as pd
from process_data import read_files
import glob

filenames = [i for i in glob.glob("/home/csdog/dataset/cleaned-limited/test.csv")]
print(filenames)

for file in filenames:
  print("------------------------------------------------------------------------")
  print("Filename: " + file)
  df = read_files([file], clean_data=False)
  df = df.groupby('Label').count()

  print(df)
  print("------------------------------------------------------------------------")

