from morfeus.conformer import ConformerEnsemble
from openbabel import openbabel

class ConformerOpenbabel: 
    def __init__(self, mol_file, metal_symbol):
        self.mol_file = mol_file
        self.metal_symbol = metal_symbol
    
    def get_ob_mol(self):
        mol = openbabel.OBMol()
        mol_reader = openbabel.OBConversion()
        mol_reader.SetInFormat("mol")
        
        if mol_reader.ReadFile(mol, self.mol_file):
            return mol
        else:
            print(f"Error reading molecule from {self.mol_file}")
            return None
    
    def change_metal_charge(self, new_charge):
        obmol = self.get_ob_mol()
        for atom in openbabel.OBMolAtomIter(obmol):
            atomic_num = atom.GetAtomicNum()
            symbol = openbabel.GetSymbol(atomic_num)
            if symbol == self.metal_symbol:
                atom.SetFormalCharge(new_charge)
    
    def get_ce(self, new_charge, prune_rmsd, prune_enantiomers, prune_energy):
        mol = self.change_metal_charge(new_charge)
        ce = ConformerEnsemble.from_ob_ff(mol, generate_rdkit_mol=True)
        if prune_enantiomers:
            ce.prune_enantiomers(keep="original")
        if prune_rmsd:
            ce.prune_rmsd()
        if prune_energy:
            ce.prune_energy()
        if ce is None:
            raise ValueError("no conformer ensemble object")
        return ce



if __name__ == "__main__":
    conformers = ConformerOpenbabel(f"C:\\Users\\finta\\Documents\\tu_delft\\MEP\\CREST\\mol_files\\L1.mol", metal_symbol= 'Rh')
    conformers.get_ce(1, True, False, False)