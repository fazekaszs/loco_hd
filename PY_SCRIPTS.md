# Running the scripts in the `python_codes` directory

The `python_codes` directory contains the scripts necessary to reproduce the experiments described in the article.
These scripts will accept data from a `data_sources` directory in the project root (i.e. `loco_hd/data_sources`) and
will write the output files into a `workdir` directory, also in the project root. These directories
should be __manually created__! Some scripts are going to need other manual directory creations too.
Note, that most of these scripts are dependent on other Python3 libraries. The ones that output images
will definitely require __Matplotlib__! All py files are "argument-less" runfiles, so running them only
takes this:

```bash
python3 [script filename]
```

## Experiment 1: `test_integrators.py`

(_The output of this experiment was not described in the article._)

Create a directory in `workdir` with the name of `integrator_tests`.
Running this script will output images for different weight functions and parametrizations.
In the pictures the PDF and CDF of these functions will be shown.
Modify the `param_sets` variable if you want to try out additional weight functions.

<p align="middle">
    <img src="https://github.com/fazekaszs/loco_hd/blob/master/images/test_integrators.png" alt="test_integrators" height=500">
    <br/>
    An example output for the test_integrators.py script.
</p>

__Expected runtime:__ few seconds

__Input files needed:__ none

## Experiment 2: `simple_test.py`

(_The output of this experiment was not described in the article._)

This will test the behavior of the LoCoHD metric upon perturbations of a randomly aranged and
randomly labelled point set. The original 3D points are stored in the `points` variable and are
labelled according to `sequence1`. The primitive types in `sequence1` are sampled from the
list `categories`. The different perturbation amounts are stored in the list `deltas`. Vectors
with lengths of different values from `deltas` and random orientations are added to each point
(this is the perturbation). Then, the LoCoHD score between each point and it's perturbed pair
is calculated and plotted. The choice of sequences can be set differently for 
different experiments:

- `sequence1` and `sequence2` can be set to the same value:

```python
sequence1 = np.random.choice(categories, size=n_of_points)
...
sequence2 = np.copy(sequence1)
```
In this case, we expect 0 LoCoHD scores when delta = 0 (neither the point positions,
nor the point primitive types are modified).

- `sequence2` can be resampled from the same primitive type pool:

```python
sequence1 = np.random.choice(categories, size=n_of_points)
...
sequence2 = np.random.choice(categories, size=n_of_points)
```
Now, even at delta = 0 we expect LoCoHD scores that are greater than 0,
since the primitive types are resampled.

- `sequence2` can be resampled from an intersecting primitive type pool:

```python
sequence1 = np.random.choice(categories[:4], size=n_of_points)
...
sequence2 = np.random.choice(categories[2:], size=n_of_points)
```
Even greater LoCoHD scores. There are primitive types that do not occur in the
other point cloud.

- `sequence2` can be resampled from a disjoint primitive type pool:

```python
sequence1 = np.random.choice(categories[:3], size=n_of_points)
...
sequence2 = np.random.choice(categories[3:], size=n_of_points)
```
In this case, we expect "all 1" LoCoHD scores at all delta values.

<p align="middle">
    <img src="https://github.com/fazekaszs/loco_hd/blob/master/images/simple_test.png" alt="simple_test" width=500">
    <br/>
    An example output for the simple_test.py script.
</p>

__Expected runtime:__ "instant"

__Input files needed:__ none

## Experiment 3: Contents of the casp14 directory

### Comparison of lDDT, CAD and LoCoHD scores

This pipeline will use the structures from the CASP14 or CASP15 competition,
including both experimental and predicted structures. The goal is
to assess the behavioural similarities and differences between the
lDDT, CAD and LoCoHD scores. For this, statistical descriptors and
plots are outputted.

In order to run this script batch, you should modify the constants in 
the `config.py` script:

- `CASP_VERSION` should be either `"casp14"` or `"casp15"`
- `PREDICTOR_KEY` is a CASP contestant ID (starting with TS)
- `SOURCE_DIR` is the root directory for the structure containing folders.
 Inside this folder should be a directory named predictions (see `"PRED_TAR_DIR"`)
 and a directory named targets (see `"REFS_TAR_DIR`").
- `EXTRACTOR_OUTPUT_DIR` is the working directory, where the content of the
 CASP tarfiles should be extracted (not directly, but first processed)
- `IGNORED_STRUCTURES` are structures that should not be considered for the analysis

The scripts should be run in the following order:

1. `casp14_tarfile_structure_extractor.py` extracts the structures
 from the `.tar` files as BioPython `Structure` instances, processes and filters
 them, and the results are saved as a single `.pickle` file containing a nested dictionary.
 The first keys of the dictionary are the CASP IDs of the structures,
 while the second keys are the predictor IDs ("true", "1", ... "5").
 The values are PDB formatted strings. Note, that elements inside the structures
 will be filtered out based on the content of the reference and predicted structures!
2. `casp14_ost_target_script.py` compares all predicted structures to
 the true structure using a some of the available metrics in OpenStructure
 (lDDT, GDT_TS, RMSD, CAD-score, ...).
 Saves a pickled, highly nested dictionary. Keys to this dictionary are the 
 following: </br></br>
 __KEY__ &#10122; CASP ID (a string, e.g.: `"T1024"`) &rarr; </br>
 __KEY__ &#10123; predictor number (an integer from 1 to 5) &rarr; </br>
 __KEY__ &#10124; either "per_resi" or "single" &rarr; </br>
 __KEY__ &#10125; score name (a string, e.g.: `"lddt"`, `"cad_score"` ...) &rarr; </br>
 __KEY__ &#10126; residue ID (This only exists, if the 3rd key is `"per_resi"`.
 This is a string in the format of \[chain ID\]/\[residue number\]-\[residue name\].
 For example: `"A/195-GLY"`.) &rarr; </br> 
 __VALUE__ : the corresponding score </br>
 <br> This script was tested in the __OpenStructure docker container__, but
 can be run (in theory) using the non-dockerized version.
3. `casp14_extend_with_locohd.py` extends the pickled dictionary saved from the
 previous step with LoCoHD score data.
4. `casp14_plotting.py` creates CAD-LoCoHD and lDDT-LoCoHD scatter plots.
 Also saves two 2D histograms containing the cumulative results of the scatter plots.
5. `casp14_statistics.py` creates and saves a dictionary (again, in a pickled format)
 containing statistics about the score calculations.
 A schema for the pickled dictionary can be seen below.
 Under the key `"median_summary"` the statistics for the per-residue median values can be seen.
 NDArrays contain the statistics for the different score types in the same order as the 
 score name order in `"score_names"`. The contents of `"correlation_mx_summary"` is the same, just
 with elements of the per-residue score correlation matrices.
 In `"median_gaps"` the largest gaps for the different per-residue scores are collected. Values are
 tuples containing the structure name for which the largest median gap was found, the structure 
 indices creating the gap, and the gap size itself, respectively. In `"corrmx_gaps"` it is all 
 the same, just with score-name pairs (stored as two-element frozensets).
 </br>

```
{
    "score_names": np.ndarray[str],
    
    "median_summary": {
        "min": ndarray[1, float], "max": ndarray[1, float],
        "mean": ndarray[1, float], "median": ndarray[1, float],
        "StDev": ndarray[1, float]
    },
    "correlation_mx_summary": {     
        "min": ndarray[2, float], "max": ndarray[2, float],
        "mean": ndarray[2, float], "median": ndarray[2, float],
        "StDev": ndarray[2, float]
    },
    
    "median_gaps": {
         "SCORE NAME": (
             "STRUCTURE NAME", "IDX1", "IDX2", 
             largest gap size as a single float
         ),
         ...
    },
    "corrmx_gaps": {
         frozenset({"SCORE NAME 1", "SCORE NAME 2"}): (
             "STRUCTURE NAME", "IDX1", "IDX2", 
             largest gap size as a single float
         ),
         ...     
    }
}
```

6. `casp14_compare_specific_structures.py` creates histograms and B-factor labelled
 structures for a specific predicted structure-pair. To specify the proteins to be
 compared set the constants `STRUCTURE_NAME`, `PREDICTED_SUFFIX1`
 and `PREDICTED_SUFFIX2` (the latter ones are the prediction indices).
 The constant `RESI_IDX_SHIFT` shifts the residue indices to match a certain
 numbering scheme. For example: for the structure T1064TS427 - downloaded from the CASP14 
 archive - to match the numbering in the PDB 7JTL a `RESI_IDX_SHIFT` of 15 is needed. Constants
 `MAX_LCHD` and `MAX_LDDT` set the plotting maximum for the LoCoHD and lDDT scores, respectively.

Structures can be downloaded from the 
[linked](https://doi.org/10.6084/m9.figshare.24885540.v1) 
FigShare repository or from the
[CASP14 archive](https://predictioncenter.org/download_area/CASP14/).

Example outputs for the `casp14_plotting.py` and `casp14_compare_specific_structures.py` 
scripts:

<p align="middle">
    <img src="https://github.com/fazekaszs/loco_hd/blob/master/images/lddt_T1064TS427_1.png" alt="lddt_T1064TS427_1" height=400">
    <br/>
    <img src="https://github.com/fazekaszs/loco_hd/blob/master/images/TS427_lDDT_full_hist.png" alt="TS427_lDDT_full_hist" height=400">
    <br/>
    <img src="https://github.com/fazekaszs/loco_hd/blob/master/images/T1064TS427_1_top10lchd.png" alt="T1064TS427_1_top10lchd" height=400">
    <br/>
    An example lDDT-LoCoHD scatterplot and heatmap and lDDT-LoCoHD barplot.
</p>

__Expected runtime:__ most scripts run within a few seconds or minutes at most, but `casp14_ost_target_script.py`
can take a bit more time to complete (at least inside the Docker container).

__Input files needed:__ all downloadable, see the description above

## Experiment 4: Contents of the pisces directory

This experiment consists of 3 parts:
1. downloading and normalizing PISCES filtered pdb files with `pisces_downloader.py`,
2. generating the raw data with `pisces_random_pairs.py`,
3. analyzing the raw data with `pisces_random_pairs_analyze.py`.

Additionally, `pisces_ring_analysis.py` generates RING analysis of the PISCES dataset
and fits a neural network on the environment interaction counts to predict the LoCoHD
between different environments. For further details, see the publication.

First, go to https://dunbrack.fccc.edu/pisces/ and generate a pdb file list filtered according
to your needs. Then, download the list, reference it in the `PISCES_FILENAME` variable of 
`pisces_downloader.py` (also, don't forget to set the `PISCES_DIR` variable and create the
necessary directory), and run the script. It will download all pdb files mentioned in the PISCES
list and normalize them.

Next, use `pisces_random_pairs.py` to create random residue environments between proteins and
calculate their corresponding LoCoHD values. Read through the capitalized variables in the beginning of the
script and set them according to your needs (their names are self-explanatory).

Lastly, use `pisces_random_pairs_analyze.py` to calculate the random environment pairing 
statistics. Here, you only have to set the variables `DATA_SOURCE_DIR` and `DATA_SOURCE_NAME`,
so that they aggree with the outputs of the previous script. (Right now, this script is a bit
"overcomplicated", because it uses the Welford online algorithm to calculate the global average
and variance.)

What you will get are tsv files describing the statistics of different residue-category pairs.
For example, the `charge_statistics.tsv` file will partition residue-category pair observations
into "neutral-neutral", "neutral-negative", "neutral-positive", "negative-positive", etc... 
pairs and show statistics (min, max, avg, std, etc...) related to them. You will also get a histogram
and a beta-distribution fit of the random LoCoHD scores and a `fitting_params.txt` file.

<p align="middle">
    <img src="https://github.com/fazekaszs/loco_hd/blob/master/images/exp5.png" alt="exp5" width=500">
    <br/>
    An example output for Experiment 5
</p>

__Expected runtime:__ few ten minutes (without `pisces_downloader.py`, which is
highly internet-speed dependent)

__Input files needed:__ all downloadable, see the description above

## Experiment 6: `compare_ensembles.py`

Modify the `SAVE_DIR` and `PATH_AND_NAMES` variables to your own needs.
Create the `loco_hd/workdir/prot_batch_resuls` directory.
In `PATH_AND_NAMES` you have to specify the source of the ensemble containing pdb file 
(first element of the tuple) and the tag or name of the protein that is represented by the ensemble 
(second element of the tuple). These tuples are collected into a list and are run separately. 
However, the coloring of the images outputted by this script (namely, the structure-structure comparison 
heatmaps) will depend on the global maximal LoCoHD score.

This script will output the aforementioned heatmaps and primitive atom LoCoHD scores,
RMSD vs. LoCoHD scatter plots, and B-factor labelled primitive atom pdb files.

<p align="middle">
    <img src="https://github.com/fazekaszs/loco_hd/blob/master/images/exp6_1.png" alt="exp6_1" width=400">
    <img src="https://github.com/fazekaszs/loco_hd/blob/master/images/exp6_2.png" alt="exp6_2" width=400">
    <br/>
    Example outputs for Experiment 6
</p>

__Expected runtime:__ few minutes

__Input files needed:__ all downloadable, see the article

## Experiment 7: `trajectory_analyzer.py`

Again, modify the `SOURCE_DIR` and `TARGET_DIR` variables.
The `SOURCE_DIR` directory should contain an xtc trajectory file and a tpr structure
file of an MD simulation (see the variables `TRAJECTORY_PATH` and `STRUCTURE_PATH`). Running
this script will analyze the evolution of the LoCoHD scores of individual residues.
It will output the following files:

- a LoCoHD score time dependence graphs for every residue in the protein,
- a B-factor labelled pdb file, which (in contrast to the previous B-factor labelled
  files) will label each residue according to the __standard deviation__ of its LoCoHD
  score,
- a graph showing the mean residue LoCoHD score time dependence,
- a graph showing the time evolution of the LoCoHD score vector's first principal component,
- a scatter plot showing the time evolution of the LoCoHD score vector's first and second principal components,
- the covariance matrix of the residue LoCoHD scores,
- the explained variance graph of the residue LoCoHD scores' PCA,
- a text file, containing the Sarle's bimodality coefficient of each residue's LoCoHD score.

<p align="middle">
    <img src="https://github.com/fazekaszs/loco_hd/blob/master/images/exp7_1.png" alt="exp7_1" width=400">
    <img src="https://github.com/fazekaszs/loco_hd/blob/master/images/exp7_2.png" alt="exp7_2" width=400">
    <br/>
    Example outputs for Experiment 7
</p>

__Expected runtime:__ few minutes

__Input files needed:__ due to the large size of trajectory files, this dataset is not supplied

## Experiment 8: `kras_scan.py`

(_The output of this experiment was not described in the article._)

This script scans the points around a KRas chain (PDB ID: 4obe) and compares the 
environment of these points with the LoCoHD algorithm to the environment of a chosen atom. 
Then, it will output the B-factor labelled point cloud to a pdb file. The B-factors are 
the p-values of the LoCoHD scores (the p-value comes from the distribution of LoCoHD scores
calculated in Experiment 5). The point clouds can be colored in PyMol according to their B-factors,
which results in a heatmap of environmental similarities to the chosen atom.

To modify the selected atom modify the `ref_coord` variable in the `main` function.
Right now it is set to be the O2 prime atom of the GDP molecule.

<p align="middle">
    <img src="https://github.com/fazekaszs/loco_hd/blob/master/images/exp8_1.png" alt="exp8_1" width=400">
    <img src="https://github.com/fazekaszs/loco_hd/blob/master/images/exp8_2.png" alt="exp8_2" width=400">
    <br/>
    Example outputs for Experiment 8
    <br/>
    On the left: B-factor colored point cloud (blue = low, red = high)
    <br/>
    On the right: same as on the left, but with the KRas cartoon model shown
</p>

__Expected runtime:__ few seconds

__Input files needed:__ download 4obe from [here](https://www.rcsb.org/structure/4OBE)