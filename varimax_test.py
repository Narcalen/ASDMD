import numpy as np
from rotation.varimax import normalize_saturation, varimax_angle, denormalize_saturation
import matplotlib.pyplot as plt

F = np.matrix(
    '0.830 -0.396;' +
    '0.818 -0.469;' +
    '0.777 -0.470;' +
    '0.798 -0.401;' +
    '0.786 0.500;' +
    '0.672 0.458;' +
    '0.594 0.444;' +
    '0.647 0.333')

# ======================================================================================================================
# varimax
F2, saturation = normalize_saturation(F)
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