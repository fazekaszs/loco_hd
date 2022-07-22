from loco_hd import LoCoHD
import numpy as np
import matplotlib.pyplot as plt

float_formatter = "{:.2f}".format
np.set_printoptions(formatter={"float_kind": float_formatter})

fig, ax = plt.subplots(1, 3)

n_of_elements = 100
max_deviation = 0.1

categories = ["A", "B", "C"]
params_a = np.array([1., 3., 1.])
params_b = 1. / np.array([3., 7., 10.])
lchd = LoCoHD(categories, params_a, params_b)

resi = np.random.choice(categories, n_of_elements)

coords1 = np.random.uniform(0, 10, size=(n_of_elements, 3))
coords2 = coords1 + np.random.uniform(0, max_deviation / np.sqrt(3), size=(n_of_elements, 3))

deviation = np.sqrt(np.sum((coords1 - coords2) ** 2, 1))

dmx1 = coords1[np.newaxis, :] - coords1[:, np.newaxis, ...]
dmx1 = np.sqrt(np.sum(dmx1 ** 2, axis=2))

dmx2 = coords2[np.newaxis, :] - coords2[:, np.newaxis, ...]
dmx2 = np.sqrt(np.sum(dmx2 ** 2, axis=2))

ax[0].imshow(dmx1)
ax[1].imshow(dmx2)

h_results = []

for row1, row2 in zip(dmx1, dmx2):
	
	mask1 = np.argsort(row1)
	mask2 = np.argsort(row2)
	
	result = lchd.hellinger_integral(resi[mask1], resi[mask2], row1[mask1], row2[mask2])
	
	h_results.append(result)
	
ax[2].scatter(deviation, h_results)
plt.show()
