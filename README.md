# Welcome to __LoCoHD__!

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/Rust-1.70-red)](https://www.rust-lang.org/)
[![Maturin](https://img.shields.io/badge/Maturin-0.14-green)](https://github.com/PyO3/maturin)

[![doi](https://img.shields.io/badge/doi-soon-fuchsia)]()
[![PyPI](https://img.shields.io/badge/PyPI-soon-fuchsia)]()

<p align="middle"><img src="./locohd_logo.png" alt="logo" width=500/></p>


__LoCoHD__ (_Local Composition Hellinger Distance_) is a metric for comparing protein structures. It can be used for one single structure-structure comparison, for the comparison of multiple structures inside ensembles, or for the comparison of structures inside an MD simulation trajectory. It is also a general-purpose metric for labelled point clouds with variable point counts. In contrast to 
[RMSD](https://en.wikipedia.org/wiki/Root-mean-square_deviation_of_atomic_positions), 
the [TM-score](https://en.wikipedia.org/wiki/Template_modeling_score), 
[lDDT](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3799472/), 
or [GDT_TS](https://en.wikipedia.org/wiki/Global_distance_test), 
it is based on the measurement of local composition differences, rather than of the Euclidean deviations. 

## Where can I read about it?

<p style="color:red">Still unpublished...<p>

## How can I install it?

### From PyPI

With pip, it is easy to add LoCoHD to your packages:

```bash
pip install loco-hd
```

### Building from source

To build LoCoHD from source, first you need to install [Rust]((https://www.rust-lang.org/tools/install)) to your system. You also need Python3, pip, and the package [Maturin](https://github.com/PyO3/maturin). Both Rust and Maturin can be installed with the following one-liners:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

```bash
pip install maturin
```

Next, clone the repository and enter it:

```bash
git clone https://github.com/fazekaszs/loco_hd && cd loco_hd
```

Run Maturin to install LoCoHD into your active environment:

```bash
maturin develop
```

And you are done!


## How can I use it?

LoCoHD was intended to be used within Python scripts, mostly through [BioPython](https://github.com/biopython/biopython) as the main `.pdb` file reader. It is also possible to use it with other protein/molecular structure readers, but the user has to write the appropriate parser that converts the information within the file into the information required for LoCoHD. An example for this can be found [here](./python_codes/trajectory_analyzer.py), where the structures come from a molecular dynamics trajectory and parsing is achieved by [MDAnalysis](https://github.com/MDAnalysis/mdanalysis).

<p style="color:red">Detailed description is coming soon...<p>