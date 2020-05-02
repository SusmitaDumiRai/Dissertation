import numpy as np
import pandas as pd
from process_data import read_files
import glob
import sys

loc = [r"/home/csdog/dataset/cleaned-limited/test.csv",
       r"C:\Users\kxd\Documents\Thesis_Susi\data\new\cleaned-limited\bot\*.csv"]

filenames = [i for i in glob.glob(sys.argv[1])]
print(filenames)

for file in filenames:
  print("------------------------------------------------------------------------")
  print("Filename: " + file)
  df = read_files([file], clean_data=False)
  df = df.groupby('Label').count()

  print(df)
  print("------------------------------------------------------------------------")

