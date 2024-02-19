from pathlib import Path

# Which version of CASP should be analyzed?
CASP_VERSION = "casp14"

# Some available predictor keys are the following (top 5 contestants):
# CASP14 :: AF2: TS427, BAKER: TS473, BAKER-experimental: TS403, FEIG-R2: TS480, Zhang: TS129
# CASP15 :: Yang-Server: TS229, UM-TBM: TS162, PEZYFoldings: TS278, Yang: TS439, DFolding: TS074
PREDICTOR_KEY = "TS427"

# Set atom filtering protocol.
# This determines how the true and predicted structures are brought to a unified form.
FILTER_TYPE = "strict"

# Set data sources and targets.
SOURCE_DIR = Path(f"../../data_sources/{CASP_VERSION}")
PRED_TAR_DIR = SOURCE_DIR / "predictions"
REFS_TAR_DIR = SOURCE_DIR / "targets"

_out_dir_name = f"{CASP_VERSION}_{PREDICTOR_KEY}_{FILTER_TYPE}-filter_results"
EXTRACTOR_OUTPUT_DIR = Path(f"../../workdir/{CASP_VERSION}") / _out_dir_name
