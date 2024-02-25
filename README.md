# Welcome to __LoCoHD__!

[![PyPI](https://img.shields.io/pypi/v/loco-hd)](https://pypi.org/project/loco-hd)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/Rust-1.70-red)](https://www.rust-lang.org/)
[![Maturin](https://img.shields.io/badge/Maturin-0.14-green)](https://github.com/PyO3/maturin)

[![doi](https://img.shields.io/badge/doi-soon-fuchsia)]()

<p align="middle"><img src="https://github.com/fazekaszs/loco_hd/blob/master/images/locohd_logo.png" alt="logo" width=500/></p>


__LoCoHD__ (_Local Composition Hellinger Distance_) is a metric for comparing protein structures. It can be used for one single structure-structure comparison, for the comparison of multiple structures inside ensembles, or for the comparison of structures inside an MD simulation trajectory. It is also a general-purpose metric for labelled point clouds with variable point counts. In contrast to 
[RMSD](https://en.wikipedia.org/wiki/Root-mean-square_deviation_of_atomic_positions), 
the [TM-score](https://en.wikipedia.org/wiki/Template_modeling_score), 
[lDDT](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3799472/), 
or [GDT_TS](https://en.wikipedia.org/wiki/Global_distance_test), 
it is based on the measurement of local composition differences, rather than of the Euclidean deviations.

## Where can I read about it?

This work is yet to be published in a scientific journal.

## Demo scripts

If you are interested in how to run the Python scripts that are used for the creation of the article, see
the [PY_SCRIPTS.md](PY_SCRIPTS.md) file.

## System requirements

### Software dependencies

LoCoHD is a Python3 package, so it most definitely requires Python3.
Additionally, it also needs BioPython and Numpy. It was tested with
- Python version 3.10.10,
- BioPython version 1.81,
- Numpy version 1.21.6.

Its build-dependencies are Rust (tested with `rustc` version 1.70.70) and
Maturin (tested with version 0.14.15).

Some of the scripts in `python_codes` use other packages too:
- Matplotlib version 3.7.1
- SciPy version 1.10.0
- MDAnalysis version 2.4.2
- scikit-learn version 1.2.1

The full package and scripts were tested on Linux (Pop!_OS 22.04 LTS).
Installation was tested on Pop!_OS and OpenSUSE Leap 15.3.

### Hardware requirements

No special hardware is needed to run LoCoHD. It was written and tested on a
laptop with the following specs:
- Model = Dell Latitude 5490
- RAM = 16 Gb
- Processor = Intel Core i5-8250U CPU @ 1.60GHz x 8

Since it doesn't need much RAM and CPU power to run, theoretically it can be also
ran on less capable machines. 

## How can I install it?

### Using docker

A Dockerfile is provided to install LoCoHD in a containerized manner.
Make sure that you have [docker](https://www.docker.com/) installed on your system.
Clone the GitHub repository and enter it:

```bash
git clone https://github.com/fazekaszs/loco_hd && cd loco_hd
```

Next, build the image:

```bash
docker build -t loco_hd:latest .
```

Using this way, you can either use the LoCoHD CLI from this image...: 

```bash
docker run --rm loco_hd:latest [LoCoHD arguments]
```

...or run custom scripts:

```bash
docker run --rm -v [ptscr]:/script -v [ptstr]:/structures --entrypoint python loco_hd:latest [ptscr]
```

where `[ptscr]` is the local path to the script to run (can be `$(pwd)` for example),
and `[ptstr]` is the path to the structures to be compared (probably used by the script).

<b style="color:red">Note:</b> This docker image only contains LoCoHD, BioPython, NumPy and the standard Python
library installed. Your scripts won't be able to utilize other libraries in it, like SciPy, ScikitLearn or 
Matplotlib. If you want to use these, modify the Dockerfile accordingly.

### Install the Rust compiler

If you install LoCoHD from source or using pip, you definitely need to install 
[Rust](https://www.rust-lang.org/tools/install) to your system.
To do this you can choose from several methods. Either you install Rust using the
"standard" way with the official script:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

which installs Rust globally, or if you are using an environment manager, like 
[Anaconda](https://www.anaconda.com/),
[Miniconda](https://docs.conda.io/en/latest/miniconda.html),
or [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge),
you can simply install Rust with

```bash
conda install -c conda-forge rust
```

If you are a Windows user, 
visit [this](https://forge.rust-lang.org/infra/other-installation-methods.html) link.

The overall installation time does not exceed a few minutes.

<b style="color:red">Note:</b> Maturin is a build-dependency of LoCoHD, and it won't run outside 
a virtual environment!

### From PyPI

With pip, it is easy to add LoCoHD to your packages:

```bash
pip install loco-hd
```

### Building from source

Besides Rust, you will also need Python3, pip, and the package [Maturin](https://github.com/PyO3/maturin) if you choose
building from source. Maturin can be installed with the following one-liner:


```bash
pip install maturin
```

or

```bash
conda install -c conda-forge maturin
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


## Running the Rust unit tests

Unit tests can be run with Cargo. Since this is a PyO3 project, an additional 
`--no-default-features` flag is needed:

```bash
cargo test --no-default-features
```


## How can I use it in the CLI?

Although this is __highly experimental__ yet and was not thoroughly tested, it is possible to use LoCoHD from the CLI.
Generally, you run it like this:

```bash
python -m loco_hd [LoCoHD arguments]
```

or using the containerized version:

```bash
docker run [Docker arguments] loco_hd:latest [LoCoHD arguments]
```

The required LoCoHD arguments are the following (for more information, 
use the `--help` flag.):
- `-s1` specifies the path to the first structure (pdb) file
- `-s2` specifies the path to the second structure (pdb) file
- `-pts` specifies the path to the primitive typing scheme (json) file
- `-afp` specifies the path to an anchor pairing file

The latter flag must point to a file having the following properties:
- it should be a simple text file, not a binary
- it can contain newline characters, since these will be stripped from the file (but spaces won't!)
- it must contain primitive atom __pair__ defining parts, separated by semicolons
- a primitive atom pair defining part must contain exactly two primitive atom defining part,
 separated by a single colon
- a primitive atom defining part contains a chain ID (e.g.: `A`), a residue ID (e.g.: `123-GLY`), and
 an atom set (e.g.: `CA,CB,CG`, __without__ spaces!) separated by forward-slashes

The latter one is necessary, since primitive atoms can come from multiple true atoms (like centroids 
and coarse grained atoms). Here is an example file:

```text
 /2-GLU/OE1,OE2:B/73-CYS/SG;
 /6-ARG/CZ:B/82-ILE/CB,CG1,CG2,CD1;
 /6-ARG/CZ:B/105-ARG/CZ;
 /21-ARG/CZ:B/105-ARG/CZ
```

This file specifies that a primitive atom coming from chain `" "` (note the space!), residue Glu<sup>2</sup>, 
atom set `{OE1, OE2}` should be paired up with a primitive atom coming from chain `"B"`, residue Cys<sup>73</sup>, 
atom set `{SG}`.
The environments around these anchors will be compared (along with 3 other pairings) using LoCoHD.

## How can I use it in my scripts?

LoCoHD was originally intended to be used within Python scripts (and this is still
the preferred way), mostly through [BioPython](https://github.com/biopython/biopython) as the main `.pdb` file reader. 
It is also possible to use it with other protein/molecular structure readers, 
but the user has to write the appropriate parser that converts the information 
within the file into the information required for LoCoHD. 
An example for this can be found [here](./python_codes/trajectory_analyzer.py), where the structures come from 
a molecular dynamics trajectory and parsing is achieved by [MDAnalysis](https://github.com/MDAnalysis/mdanalysis).

For the comparison of two protein structures with LoCoHD the following simple steps are necessary:

### 1. Loading the structures from pdb files

The imports defined here are necessary for the following code sections.

```python
from pathlib import Path
from Bio.PDB.PDBParser import PDBParser
from loco_hd import *

structure1 = PDBParser(QUIET=True).get_structure("s1", "path/to/structure1.pdb")
structure2 = PDBParser(QUIET=True).get_structure("s2", "path/to/structure2.pdb")
```

### 2. Selecting the primitive typing scheme

In this section, the true protein structures (with "true" atoms) are converted into primitive template structures (lists containing ```PrimitiveAtomTemplate``` instances). These serve as intermediate instances between the ```Atom``` class (from BioPython) and the ```PrimitiveAtom``` class (from loco-hd).

```python
primitive_assigner = PrimitiveAssigner(Path("path/to/primitive/typing/scheme.json"))
pra_templates1 = primitive_assigner.assign_primitive_structure(structure1)
pra_templates2 = primitive_assigner.assign_primitive_structure(structure2)
```

### 3. Selecting the anchor atoms

Here, it is assumed that the two structures contain the same number of anchor atoms and are paired in the same order. This is not necessary, since the anchor atom selection and pairing is easily customizable by just selecting the primitive atom index pairs.
In these examples it is only assumed to simplify things.

In the case, where all atoms are anchor atoms we can use:

```python
anchor_pairs = [
    (idx, idx) 
    for idx in range(len(pra_templates1))
]
```

Or if only primitive atoms with the ```"Cent"``` primitive type are anchors:

```python
anchor_pairs = [
    (idx, idx) 
    for idx, prat in enumerate(pra_templates1)
    if prat.primitive_type == "Cent"
]
```

The only important thing is that the indices inside the tuples must be valid within the first and second primitive atom (template) lists.
An example for a more complicated pairing is given here, where we only consider `PrimitiveAtom` pairs,
where one has a primitive type of `"O_neg"` and the other has a primitive type of `"C_aro"`:

```python
anchor_pairs = [
    (idx_a, idx_b)
    for idx_a, prat_a in enumerate(pra_templates1)
    if prat_a.primitive_type == "O_neg"
    for idx_b, prat_b in enumerate(pra_templates2)
    if prat_b.primitive_type == "C_aro"
]
```

### 4. Conversion of ```PrimitiveAtomTemplate``` instances to ```PrimitiveAtom``` instances

The intermediate templates are only necessary, so we can have an opportunity to set the ```tag``` field of our ```PrimitiveAtom```s. This field is used for the conditional setting of the "environment" of each anchor atom. For example, this can be used to ban homo-residue contacts, i.e. to ban a primitive atom from the environment of an anchor atom __if__ the primitive atom comes from the same residue as the anchor. For further explanation see section #5.

To do the conversion in a clean and effective manner we can define the following function:

```python
def prat_to_pra(prat: PrimitiveAtomTemplate) -> PrimitiveAtom:

    resi_id = prat.atom_source.source_residue
    resname = prat.atom_source.source_residue_name
    source = f"{resi_id[2]}/{resi_id[3][1]}-{resname}"

    return PrimitiveAtom(
        prat.primitive_type, 
        source,  # this is the tag field!
        prat.coordinates
    )
```

After this, a simple ```map``` will do:

```python
pra1 = list(map(prat_to_pra, pra_templates1))
pra2 = list(map(prat_to_pra, pra_templates2))
```

### 5. Creating the ```LoCoHD``` instance

This will create a simple LoCoHD instance that operates with a uniform weight function between 3 and 10 angströms and doesn't consider the ```tag``` field of the primitive atoms (i.e. it accepts any anchor atom - primitive atom contacts):

```python
lchd = LoCoHD(primitive_assigner.all_primitive_types)
```

To explicitly state the weight function use:

```python
w_func = WeightFunction("uniform", [3., 10.])
lchd = LoCoHD(primitive_assigner.all_primitive_types, w_func)
```

There is a collection of weight functions available.

Or to explicitly state the tag-pairing rule:

```python
w_func = WeightFunction("uniform", [3., 10.])
tag_pairing_rule = TagPairingRule({"accept_same": False})
lchd = LoCoHD(
    primitive_assigner.all_primitive_types, 
    w_func,
    tag_pairing_rule
)
```

The latter code creates a ```LoCoHD``` instance that considers the ```tag``` field and disregards primitive atoms in the environment that have the same tag as the anchor atom. 

Other tag pairing rules are also available.

Finally, the number of parallel threads LoCoHD can use can also be set as a last argument:

```python
lchd = LoCoHD(
    primitive_assigner.all_primitive_types, 
    w_func,
    tag_pairing_rule,
    4
)
```

### 6. Calculation of the LoCoHD scores

The LoCoHD class offers several methods for LoCoHD score calculation. These are the:

- ```from_anchors``` method, calculating a __single__ LoCoHD score from two anchor atom environments,
- ```from_dmxs``` method, calculating __several__ LoCoHD scores, each belonging to corresponding row-pairs of primitive atom distance-matrices,
- ```from_coords``` method, calculating __several__ LoCoHD scores from the coordinates of primitive atoms (it uses the ```from_dmxs``` method under the hood),
- ```from_primitives``` method, calculating __several__ LoCoHD scores from a list of ```PrimitiveAtom``` instances.

Most of the time the ```from_primitives``` method should be used. This is the only method that uses ```PrimitiveAtom``` instances, takes tag pairing rules into account, and speeds up calculations through the use of an upper distance cutoff for the environments.


```python
lchd_scores = lchd.from_primitives(
    pra1,
    pra2,
    anchor_pairs,
    10.  # upper distance cutoff at 10 angströms
)
```

This gives a list of LoCoHD scores (floats), each describing the environmental difference/distance/dissimilarity between two anchor atom environments. This is a score between 0 and 1, with larger values meaning greater dissimilarity.