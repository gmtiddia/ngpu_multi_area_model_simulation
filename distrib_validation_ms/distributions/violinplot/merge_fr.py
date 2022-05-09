import numpy as np
import pandas as pd
import glob

xmin=0.0
xmax=100.0

extension = "csv"
all_filenames = [i for i in glob.glob('firing_rate/*.{}'.format(extension))]
print("Merging csv...")
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
print("Actual lenght: ", len(combined_csv["fr"]))
combined_csv = combined_csv[combined_csv["fr"]>=xmin]
combined_csv = combined_csv[combined_csv["fr"]<=xmax]
print("Final lenght: ", len(combined_csv["fr"]))
print("Max: ", max(combined_csv["fr"]))
print("Min: ", min(combined_csv["fr"]))
combined_csv.to_csv( "firing_rate/firing_rate.csv", index=False, encoding='utf-8-sig')
