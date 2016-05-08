import numpy as np
import pandas as pd
from nipals.nipals import nipals1
from nipals.filereader import strip_first_col
import os
import matplotlib.pyplot as plt

delimiter = ","
input_filename = "/data_no_population.csv"
output_filename = "/output.csv"
current_dir = os.path.dirname(__file__)
input_file_path = current_dir + input_filename
output_file_path = current_dir + output_filename

# ======================================================================================================================
# read data

# numpy matrix for calculations
data = np.loadtxt(strip_first_col(input_file_path, delimiter=delimiter), delimiter=delimiter, skiprows=1, ndmin=2, dtype=float)

# panda data frame for fancy presentation
row_names = np.genfromtxt(input_file_path, delimiter=delimiter, skip_header=1, usecols={0}, dtype='str')
col_names = np.genfromtxt(input_file_path, delimiter=delimiter, dtype='str', max_rows=1)
data_frame = pd.DataFrame(data, index=row_names, columns=col_names)
# ======================================================================================================================
# standartize data

means = np.mean(data, axis=0)
# print("means=" + str(means))
# “Delta Degrees of Freedom”: the divisor used in the calculation is N - ddof,
# where N represents the number of elements. By default ddof is zero.
variance = np.var(data, axis=0, ddof=1)
stdev = np.std(data, axis=0, ddof=1)
# print("stdev=" + str(stdev))
data_std = (data - means) / stdev
# ======================================================================================================================
# invoke nipals algorithm to find weight values
# print("data_std=" + str(data_std))
data_std1 = data_std.transpose().dot(data_std)
# print("data_std1=" + str(data_std1))
t, p = nipals1(data_std)
# print("t=" + str(t))
# print("p=" + str(p))

# # check that nipals is as good as eigenvalues/eigenvectors function
w, v = np.linalg.eig(data_std.transpose().dot(data_std))
# s = t.transpose().dot(t)
# print("t.shape = " + str(t.shape))
# print("p.shape = " + str(p.shape))
# print("s.shape = " + str(s.shape))
# print("w.shape = " + str(w.shape))
# print("v = " + str(v))

# print(np.diagonal(t.transpose().dot(t))/40)
# print(str(p - v))

# ======================================================================================================================
# retrieve main components
res_data = data_std.dot(p)
# print("res_data.shape = " + str(res_data.shape))

# write to output file
res_data_frame = pd.DataFrame(res_data, index=row_names)
res_data_frame.to_csv(output_file_path, float_format="%.2f")

plt.scatter(res_data[:, 0], res_data[:, 1])
plt.plot([0, 0], [-np.amax(res_data[:, 1]), np.amax(res_data[:, 1])])
plt.plot([-np.amax(res_data[:, 0]), np.amax(res_data[:, 0])], [0, 0])
plt.axis([-np.amax(res_data[:, 0]), np.amax(res_data[:, 0]), -np.amax(res_data[:, 1]), np.amax(res_data[:, 1])])
plt.xlabel("P1")
plt.ylabel("P2")
plt.show()

plt.scatter(res_data[:, 0], res_data[:, 2])
plt.plot([0, 0], [-np.amax(res_data[:, 2]), np.amax(res_data[:, 2])])
plt.plot([-np.amax(res_data[:, 0]), np.amax(res_data[:, 0])], [0, 0])
plt.axis([-np.amax(res_data[:, 0]), np.amax(res_data[:, 0]), -np.amax(res_data[:, 2]), np.amax(res_data[:, 2])])
plt.xlabel("P1")
plt.ylabel("P3")
plt.show()

plt.scatter(res_data[:, 1], res_data[:, 2])
plt.plot([0, 0], [-np.amax(res_data[:, 2]), np.amax(res_data[:, 2])])
plt.plot([-np.amax(res_data[:, 1]), np.amax(res_data[:, 1])], [0, 0])
plt.axis([-np.amax(res_data[:, 1]), np.amax(res_data[:, 1]), -np.amax(res_data[:, 2]), np.amax(res_data[:, 2])])
plt.xlabel("P2")
plt.ylabel("P3")
plt.show()
