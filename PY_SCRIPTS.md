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
    <img src="https://github.com/fazekaszs/loco_hd/blob/master/images/exp1.png" alt="exp1" width=500">
    <br/>
    An example output for Experiment 1
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
    <img src="https://github.com/fazekaszs/loco_hd/blob/master/images/exp2.png" alt="exp2" width=500">
    <br/>
    An example output for Experiment 2
</p>

__Expected runtime:__ "instant"

__Input files needed:__ none

## Experiment 3: Contents of the casp14 directory

### Comparison of lDDT, CAD and LoCoHD scores

This pipeline will use the structures from the CASP14 competition,
including both experimental and predicted structures. The goal is
to assess the behavioural similarities and differences between the
lDDT, CAD and LoCoHD scores. For this, statistical descriptors and
plots are outputted.

The scripts should be run in the following order:

1. `casp14_tarfile_structure_extractor.py` extracts the structures
 from the `.tar` files as BioPython `Structure` instances. The results
 are saved as a single `.pickle` file containing a nested dictionary.
 The first keys of the dictionary are the CASP IDs of the structures,
 while the second keys are the predictor IDs ("true", "1", ... "5").
 The values are the BioPython structures. Note, that atoms not present 
 in the experimental structures, but are present in the predicted 
 structures are filtered out! Set the constants `PREDICTOR_KEY`,
 `TARFILE_ROOT` and `TARGET_DIR` to their appropriate values.
2. `casp14_create_data_for_ost.py` converts the previously mentioned
 pickled file. Saves a similar pickled dictionary, but values are now
 `.pdb` formatted strings. Constants to set: `PREDICTOR_KEY` and `SOURCE_DIR`.
3. `casp14_ost_target_script.py` compares all predicted structures to
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
 Only the `PREDICTOR_KEY` constant should be set.
4. `casp14_extend_with_locohd.py` extends the pickled dictionary saved from the
 previous step with LoCoHD score data. Set the constants `PREDICTOR_KEY` and `WORKDIR`.
5. `casp14_plotting.py` creates CAD-LoCoHD and lDDT-LoCoHD scatter plots.
 Also saves two 2D histograms containing the cumulative results of the scatter plots.
 Constants to set, again, are `PREDICTOR_KEY` and `WORKDIR`.
6. `casp14_statistics.py` creates and saves a dictionary (again, in a pickled format)
 containing statistics about the score calculations. The following keys are valid in the 
 saved dictionary: </br> </br>
 __KEY__ &#10122; either `"score_names"`, `"median_summary"`, `"correlation_mx_summary"`,
 `"median_gaps"` or `"corrmx_gaps"`
 &rarr; </br>
 __KEY__ &#10123; any of the following: `"min"`, `"max"`, `"mean"`, `"median"` or
 `"StDev"` (only if __KEY__ &#10122; is either `"median_summary"` or `"correlation_mx_summary"`) 
 or any of the score names / score-pair names (only if __KEY__ &#10122; is either 
 `"median_gaps"` or `"corrmx_gaps"`)
 &rarr; </br>
 __VALUE__ </br> </br>
 If __KEY__ &#10122; is `"score_names"`, then __KEY__ &#10123; does not exist and the 
 __VALUE__ is simply a numpy ndarray with `dtype=str`, containing the different score
 names (lDDT, CAD, LoCoHD). </br>
 If __KEY__ &#10122; is either `"median_summary"` or `"correlation_mx_summary"`, then
 __KEY__ &#10123; exists and the __VALUE__ is a numeric numpy ndarray, containing the
 corresponding score statistics. </br>
 If __KEY__ &#10122; is either `"median_gaps"` or `"corrmx_gaps"`, then __KEY__ &#10123; 
 is one of the score names / score-pair names (as a two-element frozenset) and __VALUE__
 is a tuple describing the largest median score- or SpR gaps found. The first element of 
 the tuple is the structure name, the next two elements are the predicted structure
 indices between 1 and 5, while the last element is the size of the gap.

__----- !! UNDER CONSTRUCTION !! -----__

This will compare two structures coming from the same CASP14 contestant for
the same target protein. The contestant (predictor) name is set by the variable
`PREDICTOR_NAME`, while the protein name is set by the variable `STRUCTURE_NAME`.
The exact predictions are set by the suffixes `PREDICTED_SUFFIX1` and `PREDICTED_SUFFIX2`,
which come after the protein and predictor name. Create the directories specified by
the `LDDT_TARS_PATH`, `PDB_DIR_PATH` and `TARGET_PATH` variables. The former two
directories should contain the necessary input files:

- `LDDT_TARS_PATH` should contain the `{STRUCTURE_NAME}.tgz` file,
- `PDB_DIR_PATH` should contain the corresponding `{STRUCTURE_NAME}{PREDICTOR_NAME}{PREDICTED_SUFFIX1}.pdb`
  and `{STRUCTURE_NAME}{PREDICTOR_NAME}{PREDICTED_SUFFIX2}.pdb` files, along with the true
  (experimental) pdb file named `{STRUCTURE_NAME}.pdb`.

These can be downloaded from the CASP14 archive:

- https://predictioncenter.org/download_area/CASP14/predictions/regular/
- https://predictioncenter.org/download_area/CASP14/results/lddt/

The outputs will be

- LoCoHD score and lDDT score B-labelled pdb files,
- three pdb files containing the primitive atoms of the true structure and the predicted structures,
- two images depicting the LoCoHD scores of the top 10 residues.

The residue numbers on the images can be shifted with the `RESI_IDX_SHIFT` integer variable,
while the maximum LoCoHD score shown can also be set by the `MAX_LCHD` variable.

<p align="middle">
    <img src="https://github.com/fazekaszs/loco_hd/blob/master/images/exp3.png" alt="exp3" width=500">
    <br/>
    An example output for Experiment 3
</p>

__Expected runtime:__ few seconds

__Input files needed:__ all downloadable, see the description above

## Experiment 4: `casp14_predictor_extractor.py` and `casp14_predictor_test.py`

This experiment consists of 2 parts:

1. extraction, selection, parsing and collection of the necessary protein structures,
   done by the script `casp14_predictor_extractor.py`, and
2. calculation of the LoCoHD scores, plotting, and statistics calculation,
   done by the script `casp14_predictor_test.py`.

Download all files from https://predictioncenter.org/download_area/CASP14/targets/ and put them in
`loco_hd/data_sources/casp14`, i.e. to the path set by the `TARFILE_ROOT` variable in
`casp14_predictor_extractor.py`. Here, also create the directory `loco_hd/workdir/casp14`
(referenced by `TARGET_DIR`) if you haven't done it already. Set the `PREDICTOR_KEY` to the 
contestant's CASP14 key you want to test. This script will output a single pickled file containing
the BioPython parsed pdb structures in a dictionary.

This file will be read by `casp14_predictor_test.py`, so running the previous script is
necessary for it to succeed. It will also use lDDT info containing tar files that can be
downloaded from one of the links in Experiment 3. Download all of these files into 
`loco_hd/data_sources/casp14/lDDTs` (corresponding variable: `LDDT_TARS_PATH`).
To download all files from here, you can use:

```bash
wget --no-parent -r https://predictioncenter.org/download_area/CASP14/results/lddt/
```

The outputs will be 
- images for all predictions, showing individual residue lDDT vs. LoCoHD scatter plots, 
  along with simple statistics belonging to the corresponding prediction,
- a full 2D histogram as a heatmap, coalescing the scatter plots into one single 2D lDDT
  vs. LoCoHD distribution,
- a markdown (md file) describing global statistics.

<p align="middle">
    <img src="https://github.com/fazekaszs/loco_hd/blob/master/images/exp4_1.png" alt="exp4_1" width=400">
    <img src="https://github.com/fazekaszs/loco_hd/blob/master/images/exp4_2.png" alt="exp4_2" width=400">
    <br/>
    Example outputs for Experiment 4
</p>

__Expected runtime:__ few minutes

__Input files needed:__ all downloadable, see the description above

## Experiment 5: `pisces_downloader.py`, `pisces_random_pairs.py` and `pisces_random_pairs_analyze.py`

This experiment consists of 3 parts:
1. downloading and normalizing PISCES filtered pdb files with `pisces_downloader.py`,
2. generating the raw data with `pisces_random_pairs.py`,
3. analyzing the raw data with `pisces_random_pairs_analyze.py`.

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