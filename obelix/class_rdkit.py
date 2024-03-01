from morfeus.conformer import ConformerEnsemble
from rdkit import Chem
from mace import ComplexFromMol



class ConformerRdkit:
    def __init__(self, mol_file, metal_symbol):
        self.mol_file = mol_file
        self.metal_symbol = metal_symbol

    
    def get_rdkit_mol(self):
        rdkit_mol = Chem.MolFromMolFile(self.mol_file, removeHs = False)
        if rdkit_mol is None:
            raise ValueError("failed to create rdkit mol object")
        return rdkit_mol
    

    def change_metal_bonds_to_dative(self):
        rdkit_mol = self.get_rdkit_mol()
        mol = Chem.RWMol(rdkit_mol)
        for bond in mol.GetBonds():

            if bond.GetBeginAtom().GetSymbol() == self.metal_symbol:

                begin_atom_idx = bond.GetBeginAtomIdx()
                end_atom_idx = bond.GetEndAtomIdx()
                mol.RemoveBond(begin_atom_idx, end_atom_idx)
                mol.AddBond(end_atom_idx, begin_atom_idx, Chem.BondType.DATIVE)

            if bond.GetEndAtom().GetSymbol() == self.metal_symbol:
                mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).SetAtomMapNum(1)  
                bond.SetBondType(Chem.BondType.DATIVE)
        if rdkit_mol is None:
            raise ValueError("failed to change metal bonds to dative")        
        return mol
    
    def change_charge(self):
        mol = self.change_metal_bonds_to_dative()
        for atom in mol.GetAtoms():
            atom_type = atom.GetSymbol() 
            
            if atom_type == self.metal_symbol:
                atom.SetFormalCharge(1)        
        return mol

    def create_conformers(self):
        mol = self.change_charge()
        X = ComplexFromMol(mol, geom = 'SP')
        Xs = X.GetStereomers(regime='CA', dropEnantiomers=True)
        for i, X in enumerate(Xs):
            X.AddConformers(numConfs=10)   
        return X       
        
        
if __name__ == "__main__":
    conformers = ConformerRdkit(f"C:\\Users\\finta\\Documents\\tu_delft\\MEP\\CREST\\mol_files\\L44.mol", metal_symbol= "Rh")
    ce = conformers.create_conformers()

