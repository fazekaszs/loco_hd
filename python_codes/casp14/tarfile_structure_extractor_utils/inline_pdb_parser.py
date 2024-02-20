import warnings

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Structure import Structure
from Bio.PDB.PDBExceptions import PDBConstructionWarning


class InLinePDBParser(PDBParser):

    def from_str(self, pdb_id: str, pdb_content: str) -> Structure:

        with warnings.catch_warnings():
            if self.QUIET:
                warnings.filterwarnings("ignore", category=PDBConstructionWarning)

            self.header = None
            self.trailer = None
            self.structure_builder.init_structure(pdb_id)

            self._parse([line + "\n" for line in pdb_content.split("\n")])

            self.structure_builder.set_header(self.header)

            structure = self.structure_builder.get_structure()

        return structure
