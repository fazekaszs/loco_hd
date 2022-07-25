from loco_hd import LoCoHD
import numpy as np
import matplotlib.pyplot as plt

float_formatter = "{:.2f}".format
np.set_printoptions(formatter={"float_kind": float_formatter})

fig, ax = plt.subplots()

n_of_elements = 1000

categories = np.array(["A", "B", "C", "D", "E", "F"])
params_a = np.array([1., 3., 1.])
params_b = 1. / np.array([3., 7., 10.])
lchd = LoCoHD(categories, params_a, params_b)

resi = np.random.choice(categories, n_of_elements)

coords1 = np.random.uniform(0, 10, size=(n_of_elements, 3))
dmx1 = coords1[np.newaxis, :] - coords1[:, np.newaxis, ...]
dmx1 = np.sqrt(np.sum(dmx1 ** 2, axis=2))

coords2 = np.copy(coords1)

# for idx in range(1000):

coords2 += np.random.normal(0, 0.5, size=coords2.shape)

deviation = np.sqrt(np.sum((coords1 - coords2) ** 2, 1))

dmx2 = coords2[np.newaxis, :] - coords2[:, np.newaxis, ...]
dmx2 = np.sqrt(np.sum(dmx2 ** 2, axis=2))

h_results = lchd.compare_structures(resi, resi, dmx1, dmx2)

# h_results = []

# for row1, row2 in zip(dmx1, dmx2):
	
# 	mask1 = np.argsort(row1)
# 	mask2 = np.argsort(row2)
	
# 	result = lchd.hellinger_integral(resi[mask1], resi[mask2], row1[mask1], row2[mask2])
	
# 	h_results.append(result)

ax.cla()
ax.set_ylim(0, 1)
ax.scatter(deviation, h_results, alpha=0.8)
plt.show()
