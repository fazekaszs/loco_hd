# Welcome to __LoCoHD__!

[![PyPI](https://img.shields.io/pypi/v/loco-hd)](https://pypi.org/project/loco-hd)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/Rust-1.70-red)](https://www.rust-lang.org/)
[![Maturin](https://img.shields.io/badge/Maturin-0.14-green)](https://github.com/PyO3/maturin)

[![doi](https://img.shields.io/badge/doi-soon-fuchsia)]()

<p align="middle"><img src="https://github.com/fazekaszs/loco_hd/blob/master/locohd_logo.png" alt="logo" width=500/></p>


__LoCoHD__ (_Local Composition Hellinger Distance_) is a metric for comparing protein structures. It can be used for one single structure-structure comparison, for the comparison of multiple structures inside ensembles, or for the comparison of structures inside an MD simulation trajectory. It is also a general-purpose metric for labelled point clouds with variable point counts. In contrast to 
[RMSD](https://en.wikipedia.org/wiki/Root-mean-square_deviation_of_atomic_positions), 
the [TM-score](https://en.wikipedia.org/wiki/Template_modeling_score), 
[lDDT](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3799472/), 
or [GDT_TS](https://en.wikipedia.org/wiki/Global_distance_test), 
it is based on the measurement of local composition differences, rather than of the Euclidean deviations. 

## Where can I read about it?

This work is yet to be published in a scientific journal.

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


## Running the Rust unit tests

Unit tests can be run with Cargo. Since this is a PyO3 project, an additional flag is needed:

```bash
cargo test --no-default-features
```


## How can I use it?

LoCoHD was intended to be used within Python scripts, mostly through [BioPython](https://github.com/biopython/biopython) as the main `.pdb` file reader. It is also possible to use it with other protein/molecular structure readers, but the user has to write the appropriate parser that converts the information within the file into the information required for LoCoHD. An example for this can be found [here](./python_codes/trajectory_analyzer.py), where the structures come from a molecular dynamics trajectory and parsing is achieved by [MDAnalysis](https://github.com/MDAnalysis/mdanalysis).

For the comparison of two protein structures with LoCoHD the following simple steps are necessary:

### 1. Loading the structures from pdb files

```python
# These imports are necessary for the union of the sections!
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

Here, it is assumed that the two structures contain the same number of anchor atoms and are paired in the same order. This is not necessary, since the anchor atom selection and pairing is easily customizable by just selecting the primitive atom index pairs. In these example it is only assumed to simplify things.

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
    if atom.primitive_type == "Cent"
]
```

The only important thing is that the indices inside the tuples must be valid within the first and second primitive atom (template) lists.

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