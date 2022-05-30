import numpy as np
import pandas as pd
import glob

xmin=-0.05
xmax=0.2

extension = "csv"
all_filenames = [i for i in glob.glob('correlation/*.{}'.format(extension))]
print("Merging csv...")
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
print("Actual lenght: ", len(combined_csv["corr"]))
combined_csv = combined_csv[combined_csv["corr"]>=xmin]
combined_csv = combined_csv[combined_csv["corr"]<=xmax]
print("Final lenght: ", len(combined_csv["corr"]))
print("Max: ", max(combined_csv["corr"]))
print("Min: ", min(combined_csv["corr"]))
combined_csv.to_csv( "correlation/correlation.csv", index=False, encoding='utf-8-sig')
