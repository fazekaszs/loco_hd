# Order to run the CASP14 analyzing LoCoHD scripts

## lDDT and CAD vs. LoCoHD score pipeline

1. `casp14_tarfile_structure_extractor.py`: extract the structures
 from the `.tar` files as BioPython Structure instances. The results
 are saved as a single `.pickle` file containing a nested dictionary.
 The first keys of the dictionary are the CASP IDs of the structures,
 while the second keys are the predictor IDs ("true", "1", ... "5").
 The values are the BioPython structures.
2. `casp14_create_data_for_ost.py`: convert the previously mentioned
 pickled file. Values are now `pdb` formatted strings.
3. `casp14_ost_target_script.py`: compare all predicted structures to
 the true structure using a myriad of metrics (lddt, gdtts, rmsd, 
 cad_score, ...). Save a pickled nested dictionary. Keys are the 
 following: </br>
 1 - CASP ID </br>
 2 - predictor number (1...5) </br>
 3 - either "per_resi" or "single" </br>
 4 - score name </br>
 5 - residue ID (__only exists, if the 3rd key is "per_resi"__) </br>
 The value is the corresponding score.
 This script can only be run in the __OpenStructure docker 
 container__!
4. `casp14_extend_with_locohd.py`
5. `casp14_plotting.py`
6. `casp14_statistics.py`

## Standalones

- `casp14_compare_specific_structures.py`