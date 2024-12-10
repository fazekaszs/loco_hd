from pathlib import Path

# Which version of CASP should be analyzed?
CASP_VERSION = "casp15"

# Some available predictor keys are the following (top 5 contestants):
# CASP14 :: AF2: TS427, BAKER: TS473, BAKER-experimental: TS403, FEIG-R2: TS480, Zhang: TS129
# CASP15 :: Yang-Server: TS229, UM-TBM: TS162, PEZYFoldings: TS278, Yang: TS439, DFolding: TS074
PREDICTOR_KEY = "TS278"

# Set data sources and targets.
SOURCE_DIR = Path(f"../../data_sources/{CASP_VERSION}")
PRED_TAR_DIR = SOURCE_DIR / "predictions"
REFS_TAR_DIR = SOURCE_DIR / "targets"

_out_dir_name = f"{PREDICTOR_KEY}_results"
EXTRACTOR_OUTPUT_DIR = Path(f"../../workdir/{CASP_VERSION}") / _out_dir_name

# To leave out structures from the analysis.
# IGNORED_STRUCTURES = set()
IGNORED_STRUCTURES = {
    # Results in an OST error:
    # Residue numbers in each target chain must be strictly increasing if resnum_alignments are enabled
    "H1111",
} if CASP_VERSION == "casp15" else {}
