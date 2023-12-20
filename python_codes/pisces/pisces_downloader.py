import time
import os

from urllib3 import PoolManager
from pathlib import Path
from typing import Union

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Structure import Structure
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue, DisorderedResidue
from Bio.PDB.Atom import Atom, DisorderedAtom
from Bio.PDB.PDBIO import PDBIO

ALLOWED_ELEMENTS = ["C", "S", "N", "O"]
ALLOWED_ALTLOC = [" ", "A"]
RESI_TLCS = [
    "GLY", "ALA", "VAL", "ILE", "LEU",
    "PHE", "SER", "THR", "TYR", "ASP",
    "GLU", "ASN", "GLN", "CYS", "MET",
    "PRO", "LYS", "ARG", "TRP", "HIS"
]
RESI_OLCS = [
    "G", "A", "V", "I", "L",
    "F", "S", "T", "Y", "D",
    "E", "N", "Q", "C", "M",
    "P", "K", "R", "W", "H"
]
RESI_ATOM_COUNTS = [
    4, 5, 7, 8, 8,
    11, 6, 7, 12, 8,
    9, 8, 9, 6, 8,
    7, 9, 11, 14, 10
]
PISCES_DIR = Path("../../data_sources/pisces")
PISCES_FILENAME = "cullpdb_pc25.0_res0.0-2.0_noBrks_noDsdr_len40-300_R0.25_Xray+Nmr_d2022_02_21_chains3501"


def download_pisces(pisces_source: Path, file_target: Path):

    # Open the PISCES file.
    with open(pisces_source, "r") as f:
        pdb_list = f.read()

    # Filter out empty and commented lines.
    # The first line contains the column names, so remove that.
    pisces_lines = list(filter(lambda x: not len(x) == 0 and x[0] != "#", pdb_list.split("\n")))
    pisces_lines = pisces_lines[1:]

    # Extracting the pdb_ids and chain_ids from the PISCES file lines.
    pdb_and_chain_ids = dict()
    for line in pisces_lines:
        if line[:4] not in pdb_and_chain_ids:
            pdb_and_chain_ids[line[:4]] = line[4]
        else:
            pdb_and_chain_ids[line[:4]] += line[4]

    # If the file_target directory does not exist, make it.
    if not os.path.isdir(file_target):
        os.mkdir(file_target)

    avg_size = 0.
    failed = 0

    start_time = time.time()
    for idx, pdb_id in enumerate(pdb_and_chain_ids):

        current_pdb_path = file_target / f"{pdb_id}.pdb"

        try:

            # Download the .pdb file.
            url = f"http://files.rcsb.org/download/{pdb_id}.pdb"
            file = PoolManager().request("GET", url, preload_content=False).data.decode("ascii")

            with open(current_pdb_path, "w+") as f:
                f.write(file)

            print(f"{pdb_id} downloaded!")

        except Exception as e:

            print(f"{pdb_id} download failed with message \"{e}\"!")
            failed += 1
            continue

        protein: Structure = PDBParser(QUIET=True).get_structure(pdb_id, str(current_pdb_path))

        # Model number validation.
        if len(protein) == 0:

            print(f"{pdb_id} contains no models!")
            failed += 1
            os.remove(current_pdb_path)
            continue

        # Removing unnecessary models.
        for model_idx in range(len(protein) - 1, 1, -1):
            protein.detach_child(model_idx)

        # Searching for disordered residues.
        residue: Union[Residue, DisorderedResidue]
        disordered_resi = list()
        for residue in protein[0].get_residues():

            if type(residue) is DisorderedResidue:
                disordered_resi.append(residue.full_id)

        # Replacing disordered residues.
        for _, _, chain_id, resi_id in disordered_resi:

            current_chain: Chain = protein[0][chain_id]
            correct_resi_name: str = list(current_chain[resi_id].child_dict.keys())[0]
            # Direct indexing of a DisorderedResidue references atoms, so .child_dict is necessary:
            correct_resi: Residue = current_chain[resi_id].child_dict[correct_resi_name]
            resi_idx = current_chain.child_list.index(current_chain[resi_id])

            current_chain.child_list[resi_idx] = correct_resi  # replace in parent's child list
            current_chain.child_dict[resi_id] = correct_resi  # replace in parent's child dict
            correct_resi.set_parent(current_chain)  # replace child's parent

        del disordered_resi

        # Searching for banned elements and disordered atoms.
        atom: Union[Atom, DisorderedAtom]
        atoms_to_remove = list()
        disordered_atoms = list()
        for atom in protein[0].get_atoms():

            current_full_id = list(atom.get_full_id())  # Convert tuple to list.
            current_full_id[-1] = current_full_id[-1][0]  # Remove altloc id.

            if atom.element not in ALLOWED_ELEMENTS:
                atoms_to_remove.append(current_full_id)
            elif type(atom) is DisorderedAtom:
                disordered_atoms.append(current_full_id)

        # Removing banned elements.
        for _, _, chain_id, resi_id, atom_id in atoms_to_remove:

            current_resi: Residue = protein[0][chain_id][resi_id]
            current_resi.detach_child(atom_id)

        del atoms_to_remove

        # Replacing disordered atoms with atoms.
        for _, _, chain_id, resi_id, atom_id in disordered_atoms:

            current_resi: Residue = protein[0][chain_id][resi_id]
            correct_atom: Atom = list(current_resi[atom_id].child_dict.values())[0]
            atom_idx = current_resi.child_list.index(current_resi[atom_id])

            current_resi.child_list[atom_idx] = correct_atom  # replace in parent's child list
            current_resi.child_dict[atom_id] = correct_atom  # replace in parent's child dictionary
            correct_atom.set_parent(current_resi)  # replace child's parent

            correct_atom.disordered_flag = 0  # set the atom's disordered flag to false
            current_resi.disordered = 0  # set the residue's disordered flag to false

        del disordered_atoms

        # Removing invalid residues.
        chain: Chain
        chains_to_remove = list()
        for chain_idx, chain in enumerate(protein[0].child_list):

            residue: Residue
            residues_to_remove = list()

            for residue in chain.child_list:

                # Assert correct residue naming.
                if residue.resname not in RESI_TLCS:
                    residues_to_remove.append(residue.get_id())

                # Assert correct atom count in residue.
                elif len(residue) != RESI_ATOM_COUNTS[RESI_TLCS.index(residue.resname)]:
                    residues_to_remove.append(residue.get_id())

            for resi_id in residues_to_remove:
                chain.detach_child(resi_id)

            if len(chain) == 0:
                chains_to_remove.append(chain.get_id())

        for chain_id in chains_to_remove:
            protein[0].detach_child(chain_id)

        del chains_to_remove

        # If no chains are left, delete the protein.
        if len(protein[0]) == 0:

            print(f"{pdb_id} lost all chains during cleanup!")
            failed += 1
            os.remove(current_pdb_path)
            continue

        pdb_io = PDBIO()
        pdb_io.set_structure(protein)
        pdb_io.save(str(current_pdb_path))

        # Get the file size.
        current_size = os.path.getsize(current_pdb_path)
        avg_size = (idx * avg_size + current_size) / (idx + 1)

        estimated_total_size = avg_size * len(pdb_and_chain_ids) / (1024 ** 3)
        print(f"Estimated total size: {estimated_total_size:6.3f} Gb")

        # Calculate ETA.
        current_time = time.time()
        current_eta = (current_time - start_time) * (len(pdb_and_chain_ids) - idx - 1) / (60. * (idx + 1))
        print(f"Estimated time left: {current_eta:6.3f} min.")

        # Calculate percentage.
        done_percent = 100. * (idx + 1) / len(pdb_and_chain_ids)
        print(f"Done: {idx + 1}/{len(pdb_and_chain_ids)} ({done_percent:4.3f}%)")

        # Failed.
        print(f"Failed: {failed}")


def main():
    download_pisces(PISCES_DIR / PISCES_FILENAME, PISCES_DIR)


if __name__ == "__main__":
    main()
