import csv
import glob

exclude = "merged_csv.csv"
csv_headers = {}

filenames = glob.glob("*.csv")

# print("filenames: %s" % filenames)
for f in filenames:
  # print("f: %s" % f)
  if f != exclude:
    open_f = open(f, "r")
    reader = csv.reader(open_f)
    header = next(reader, None)
    # print("header: %s" % header)
    csv_headers[f] = header
    open_f.close()


for k, v in csv_headers.items():
  l = len(v)
  print("Key: {0}, Length: {1}".format(k, l))
  # print("Values: %s" % v)
