import pickle
import numpy as np
import math
from scipy import stats
from pathlib import Path


def main():

    # data_source = Path("workdir/pisces/results_uniform-3-10_only-hetero-contacts.pickle")
    # data_source = Path("workdir/pisces/results_uniform-3-10_all-contacts.pickle")
    data_source = Path("workdir/pisces/results_kumaraswamy-3-10-2-5_only-hetero-contacts.pickle")

    with open(data_source, "rb") as f:
        data = pickle.load(f)

    print(f"Number of samples: {len(data)}")

    resi_stat = dict()
    for element in data:

        key = [element[0][-3:], element[1][-3:]]
        key = sorted(key)
        key = tuple(key)

        if key in resi_stat:
            resi_stat[key].append(element[2])
        else:
            resi_stat[key] = [element[2], ]

    for key in resi_stat:

        sample_mean = np.mean(resi_stat[key])
        sample_std = np.std(resi_stat[key])
        sample_conf_int = stats.t.ppf(0.95, len(resi_stat[key]) - 2) * sample_std / math.sqrt(len(resi_stat[key]))

        resi_stat[key] = (sample_mean, sample_std, sample_conf_int)

    resi_pairs = list(resi_stat.keys())
    resi_pairs = sorted(resi_pairs, key=lambda x: resi_stat[x][0])

    for key in resi_pairs:
        print(f"{key[0]}-{key[1]}: {resi_stat[key][0]:.2%} +/- {resi_stat[key][2]:.2%}")


if __name__ == "__main__":
    main()
