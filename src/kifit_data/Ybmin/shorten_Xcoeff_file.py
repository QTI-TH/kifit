import os
import numpy as np

elemid = "Ybmin"
file_type = "Xcoeffs"

file_name = file_type + "_" + elemid + ".dat"
new_file_name = file_type + "_" + elemid + "_new.dat"

file_path = os.path.join(file_name)

data = np.loadtxt(file_path)

newdata = data[1::50]

new_file_path = os.path.join(new_file_name)

np.savetxt(new_file_path, newdata)
