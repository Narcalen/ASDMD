import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from factors.centroid import reflect, centroid
from nipals.filereader import strip_first_col
from rotation.angle import angle_between
from rotation.varimax import normalize_saturation, varimax_angle, denormalize_saturation

delimiter = ","
input_filename = "/data_no_population.csv"
output_filename = "/output.csv"
current_dir = os.path.dirname(__file__)
input_file_path = current_dir + input_filename
output_file_path = current_dir + output_filename
factors_num = 2
np.set_printoptions(precision=4, suppress=True)
# ======================================================================================================================
# read data

# numpy matrix for calculations
input_data = np.loadtxt(strip_first_col(input_file_path, delimiter=delimiter), delimiter=delimiter, skiprows=1, ndmin=2,
                        dtype=float)

# panda data frame for fancy presentation
row_names = np.genfromtxt(input_file_path, delimiter=delimiter, skip_header=1, usecols={0}, dtype='str')
col_names = np.genfromtxt(input_file_path, delimiter=delimiter, dtype='str', max_rows=1)
data_frame = pd.DataFrame(input_data, index=row_names, columns=col_names)
# ======================================================================================================================
# standartize data

means = np.mean(input_data, axis=0)
# print("means=" + str(means))
# “Delta Degrees of Freedom”: the divisor used in the calculation is N - ddof,
# where N represents the number of elements. By default ddof is zero.
variance = np.var(input_data, axis=0, ddof=1)
stdev = np.std(input_data, axis=0, ddof=1)
# print("stdev=" + str(stdev))
data_std = (input_data - means) / stdev

# ======================================================================================================================
# get correlation matrix
correlation = np.corrcoef(data_std.transpose())
correlation = np.matrix(
    '1 0.10 0.75 0.20;' +
    '0.10 1 0.10 0.85;' +
    '0.75 0.10 1 0.05;' +
    '0.20 0.85 0.05 1'
)
# ======================================================================================================================
# change 1 on main diagonal to h^2
data = correlation.__copy__()
for i in range(0, data.shape[0]):
    data[i, i] = 0
    data[i, i] = np.amax(data[:, i])
print("input data")
print(str(data))

print("input data shape")
print(str(data.shape))

# data = np.matrix(
#     '0.8200 0.7250 0.6310 0.1250 0.1600 0.0545;' +
#     '0.7250 0.6425 0.5650 0.0080 0.1150 0.0650;' +
#     '0.6310 0.5650 0.5000 0.1150 0.1400 0.0850;' +
#     '0.1250 0.0080 0.1150 0.6425 0.5650 0.4025;' +
#     '0.1600 0.1150 0.1400 0.5650 0.5000 0.3550;' +
#     '0.0545 0.0650 0.0850 0.4025 0.3550 0.2525')

# ======================================================================================================================
# centroid methods
F = np.empty(shape=(data.shape[0], factors_num))
output = np.empty(shape=(data.shape[0], factors_num + 2))

RH = data.__copy__()
signs = np.zeros(shape=data.shape[0])
for i in range(0, factors_num):
    print("RH")
    print(str(RH))
    F[:, i] = centroid(RH)
    F[:, i] *= (-1) ** signs
    print("F" + str(i + 1))
    print(str(F[:, i]))

    RH = RH - F[:, i][:, None].dot(F[:, i][:, None].transpose())
    print("RH before reflect")
    print(str(RH))
    RH, signs = reflect(RH)
    output[:, i] = F[:, i].transpose()
    print("RH after reflect")
    print(str(RH))
    if (np.diag(RH) < 0).sum() > 0:
        for j in range(0, RH.shape[0]):
            RH[j, j] = 0
            RH[j, j] = np.amax(RH[:, j])

# ======================================================================================================================
# parse results

output[:, 2] = (np.square(F[:, 0]) + np.square(F[:, 1])).transpose()
output[:, 3] = 1 - output[:, 2]
row_names = ['z%i' % i for i in range(data.shape[0])]
col_names = ['F%i' % i for i in range(factors_num)]
col_names.append('h^2')
col_names.append('d^2')

print("result")
res_data_frame = pd.DataFrame(output, index=row_names, columns=col_names)
print(str(res_data_frame))

# ======================================================================================================================
# plot results

print("data to plot: " + str(F))
plt.scatter(F[:, 0], F[:, 1])
plt.plot([0, 0], [-1, 1])
plt.plot([-1, 1], [0, 0])
plt.axis([-1, 1, -1, 1])
plt.xlabel("F1")
plt.ylabel("F2")
plt.show()

# ======================================================================================================================
# factors rotation

position = np.argmin(np.abs(F[:, 0]))
point = F[position]
print("point = " + str(point))
if point[1] > 0:
    angle = angle_between((0, 1), point)
    rotation_matrix = np.matrix([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
else:
    angle = angle_between((1, 0), point)
    rotation_matrix = np.matrix([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
if point[0] > 0:
    rotation_matrix[0, 1] *= -1
    rotation_matrix[1, 0] *= -1
print("rotation matrix: " + str(rotation_matrix))

F1 = F.dot(rotation_matrix)
print("data to plot: " + str(F1))
plt.scatter(F1[:, 0], F1[:, 1])
plt.plot([0, 0], [-1, 1])
plt.plot([-1, 1], [0, 0])
plt.axis([-1, 1, -1, 1])
plt.xlabel("F1_rotated")
plt.ylabel("F2_rotated")
plt.show()

# ======================================================================================================================
# varimax
F2, saturation = normalize_saturation(F)
F2 = F2.transpose()
saturation = saturation.transpose()
print("normalized data: " + str(F2))
angle = varimax_angle(F2)
rotation_matrix = np.matrix([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
print("rotation matrix: " + str(rotation_matrix))
F2 = F2.dot(rotation_matrix)
F2 = denormalize_saturation(F2, saturation)
print("data to plot - after varimax rotation: " + str(F2))
plt.scatter(F2[:, 0], F2[:, 1])
plt.plot([0, 0], [-1, 1])
plt.plot([-1, 1], [0, 0])
plt.axis([-1, 1, -1, 1])
plt.xlabel("F1_varimax")
plt.ylabel("F2_varimax")
plt.show()