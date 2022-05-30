import numpy as np
import pandas as pd
import glob

xmin=0.0
xmax=5.0

extension = "csv"
all_filenames = [i for i in glob.glob('cv_isi/*.{}'.format(extension))]
print("Merging csv...")
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
print("Actual lenght: ", len(combined_csv["cv_isi"]))
combined_csv = combined_csv[combined_csv["cv_isi"]>=xmin]
combined_csv = combined_csv[combined_csv["cv_isi"]<=xmax]
print("Final lenght: ", len(combined_csv["cv_isi"]))
print("Max: ", max(combined_csv["cv_isi"]))
print("Min: ", min(combined_csv["cv_isi"]))
combined_csv.to_csv( "cv_isi/cv_isi.csv", index=False, encoding='utf-8-sig')
