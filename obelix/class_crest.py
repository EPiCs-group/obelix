from morfeus.conformer import ConformerEnsemble
from rdkit import Chem
import numpy as np
import csv

class ConformerCrest:
    def __init__(self, mol_file, crest_folder, xyz_file = None):
        self.mol_file = mol_file
        self.crest_folder = crest_folder
        self.xyz_file = xyz_file
        

    def get_rdkit_mol(self):
        rdkit_mol = Chem.MolFromMolFile(self.mol_file, removeHs = False)
        if rdkit_mol is None:
            raise ValueError("failed to create rdkit mol object")
        return rdkit_mol

    def get_connectivity_matrix(self):
        if self.get_rdkit_mol is not None:
            rdkit_mol = self.get_rdkit_mol()
            num_atoms = rdkit_mol.GetNumAtoms()
            connectivity_matrix = [[0] * num_atoms for _ in range(num_atoms)]

            for bond in rdkit_mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                order = bond.GetBondTypeAsDouble()
                connectivity_matrix[i][j] = connectivity_matrix[j][i] = order
        if connectivity_matrix is None:
            raise ValueError("failed to create connectivity matrix")
        return connectivity_matrix
    
    def get_ce(self, prune_rmsd, prune_enantiomers, prune_energy):
        ce = ConformerEnsemble.from_crest(self.crest_folder)
        ce.sort()
        ce.connectivity_matrix = self.get_connectivity_matrix()
        ce.generate_mol()
        if prune_enantiomers:
            ce.prune_enantiomers(keep="original")
        if prune_rmsd:
            ce.prune_rmsd()
        if prune_energy:
            ce.prune_energy()
        if ce is None:
            raise ValueError("no conformer ensemble object")
        return ce
    
    def get_absolute_energies(self, write_csv = False, csv_filepath = None):
        ce = self.get_ce(False, True, False)
        rel_en = np.array(ce.get_energies())
        with open(self.xyz_file, "r") as f:
            lines = f.readlines()
        initial_en = float(lines[1])
        abs_en = rel_en + initial_en

        #write csv file from it
        if write_csv:
            with open(csv_filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Energy'])  
                for energy in abs_en:
                    writer.writerow([energy])        
        return abs_en
    
    
        
             
if __name__ == "__main__":
    conformers = ConformerCrest(f"C:\\Users\\finta\\Documents\\tu_delft\\MEP\\CREST\\mol_files\\L191.mol", f"C:\\Users\\finta\\Documents\\tu_delft\\MEP\\CREST\\Output\\L191", f"C:\\Users\\finta\\Documents\\tu_delft\\MEP\\CREST\\Output\\L191\\crest_conformers.xyz")
    abs_en = conformers.get_absolute_energies(write_csv = True, csv_filepath = f"C:\\Users\\finta\\Documents\\tu_delft\\MEP\\CREST\\Absolute_energies\\abs_en_ceL191.csv")
    print(abs_en)
