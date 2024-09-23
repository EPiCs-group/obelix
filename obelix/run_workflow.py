from chemspax.main import main as chemspax_main
from chemspax.utilities import *
import os
import glob
import pandas as pd
from obelix.tools.utilities import *
from obelix.descriptor_calculator import Descriptors
import shutil
from tqdm import tqdm
import mace
import subprocess
import morfeus as mf
from morfeus.conformer import ConformerEnsemble
from morfeus import BiteAngle, ConeAngle, BuriedVolume, Dispersion, SASA, read_xyz
# from xtb.utils import get_method
import pandas as pd
from openbabel import openbabel
from obelix.molecular_graph import molecular_graph
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.info')


class MACE:   
    def __init__(self, bidentate, CA, name_of_xyz):
        # Declare instances of the class (a.k.a self variable)
        self.CA = CA
        self.name_of_xyz = name_of_xyz
        self.bidentate = bidentate

    @staticmethod
    def find_bidentate(mol, metal_atom_atomic_number):
        """Find indices of bidentate ligand if functions implemented in MACE class are not working or if
        structure was made manually and there is no mapped SMILES string

        Args:
            mol: RDKit mol object
            metal_atom_atomic_number:

        Returns: list of bidentate atom indices

        """
        bidentate_idxs = []
        try:
            # find bidentate indices for first conformer, they should be the same for all conformers
            # in case of SP we only need to find metal atom and neighbours since only one cycle with metal center is possible
            metal_atom = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == metal_atom_atomic_number][0]
            metal_bonds = [bond for bond in metal_atom.GetBonds()]
            for bond in metal_bonds:
                # atoms that are connected to the metal and in the bidentate cycle are the bidentate atoms
                idx1, idx2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                bidentate_idxs.append(idx1) if idx1 != metal_atom.GetIdx() else bidentate_idxs.append(idx2)

        except Exception as e:
            # ToDo: add logging
            # go on to next conformer and try again
            return None

        return bidentate_idxs[0], metal_atom.GetIdX(), bidentate_idxs[1]

    def generate_complex_SP_xyz(self, auxiliary_ligands = []):
        geom = 'SP'
        ligands = [self.bidentate]
        ligands.extend(auxiliary_ligands)
        core = mace.ComplexFromLigands(ligands, self.CA, geom)
        Xs = core.GetStereomers(regime='all', dropEnantiomers=True)

        # bidentate_idxs = []  # the final bidentate cycle indices
        for i, X in enumerate(Xs):
            X.AddConformers(numConfs=10)   
            X.ToXYZ(self.CA + '_' + '{}_{}.xyz'.format(self.name_of_xyz, i), confId='min')
        ligands.remove(self.bidentate)
        #     if len(bidentate_idxs) == 0:
        #         try:
        #             # find bidentate indices for first conformer, they should be the same for all conformers
        #             # in case of SP we only need to find metal atom and neighbours since only one cycle with metal center is possible
        #             atomic_number_metal_center = Chem.MolFromSmiles(self.CA).GetAtomWithIdx(0).GetAtomicNum()
        #             metal_atom = [atom for atom in X.mol3D.GetAtoms() if atom.GetAtomicNum() == atomic_number_metal_center][0]
        #             metal_bonds = [bond for bond in metal_atom.GetBonds()]
        #             for bond in metal_bonds:
        #                 # atoms that are connected to the metal and in the bidentate cycle are the bidentate atoms
        #                 idx1, idx2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        #                 bidentate_idxs.append(idx1) if idx1 != metal_atom.GetIdx() else bidentate_idxs.append(idx2)

        #         except Exception as e:
        #             # ToDo: add logging
        #             # go on to next conformer and try again
        #             continue
        return len(Xs)
        # return bidentate_idxs[0], metal_atom.GetIdX(), bidentate_idxs[1]

    def generate_complex_OH_xyz(self, auxiliary_ligands = [], substrate = []):
        geom = 'OH'
        try:
            # Check if auxiliary ligands and substrate are present
            if auxiliary_ligands == [] and substrate == []:
                auxiliary_ligands = ['[H-:1]']*4
            if auxiliary_ligands == [] and substrate != []:
                auxiliary_ligands = ['[H-:1]']*3
            
            auxiliary = auxiliary_ligands
            
            if substrate == []:
                auxiliary.extend([self.bidentate])
            else:
                auxiliary.extend([self.bidentate, substrate[0]])

            core = mace.ComplexFromLigands(auxiliary, self.CA, geom)
            Xs = core.GetStereomers(regime='all', dropEnantiomers=True)

            bidentate_cycle_idxs = None  # all cycles that could possibly be the metal-bidentate cycle
            bidentate_idxs = []  # the final bidentate cycle indices
            for i, X in enumerate(Xs):
                X.AddConformers(numConfs=10)   
                X.ToXYZ(self.CA + '_' + '{}{}.xyz'.format(self.name_of_xyz, i), confId='min')
        except Exception as e:
            print('Error in MACE: ', e)

        auxiliary.remove(self.bidentate)
        auxiliary.remove(substrate[0])
        #     # find bidentate indices on the conformer if none were yet
        #     if len(bidentate_idxs) == 0:
        #         try:
        #             # get all mapped (donor) atom indices
        #             donor_atoms = [atom.GetIdx() for atom in X.mol3D.GetAtoms() if atom.GetAtomMapNum()]
        #             # smallest set of simple rings containing at least two mapped atoms
        #             # Chem.GetSSSR() is updated in rdkit 2020.09.1, versions before had to use Chem.GetSymmSSSR()
        #             atoms_of_smallest_rings = [list(idx) for idx in Chem.GetSSSR(X.mol3D)]
        #             for list_idxs in atoms_of_smallest_rings:
        #                 # if at least two mapped atoms are in the ring it's a multidentate ligand
        #                 if check_if_at_least_two_mapped_atoms_in_ring(donor_atoms, list_idxs):
        #                     bidentate_cycle_idxs = list_idxs
        #                     break

        #             # find metal atom and neighbours
        #             atomic_number_metal_center = Chem.MolFromSmiles(self.CA).GetAtomWithIdx(0).GetAtomicNum()
        #             metal_atom = [atom for atom in X.mol3D.GetAtoms() if atom.GetAtomicNum() == atomic_number_metal_center][0]
        #             metal_bonds = [bond for bond in metal_atom.GetBonds()]
        #             for bond in metal_bonds:
        #                 # atoms that are connected to the metal and in the bidentate cycle are the bidentate atoms
        #                 idx1, idx2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        #                 if idx1 == metal_atom.GetIdx() and idx2 in bidentate_cycle_idxs:
        #                     bidentate_idxs.append(idx2)
        #                 elif idx1 in bidentate_cycle_idxs and idx2 == metal_atom.GetIdx():
        #                     bidentate_idxs.append(idx1)

        #         except Exception as e:
        #             # ToDo: add logging
        #             # go on to next conformer and try again
        #             continue

        # return bidentate_idxs[0], metal_atom.GetIdX(), bidentate_idxs[1]
            

class Workflow:
    
    #### Initialize
    def __init__(self, mace_input=[], chemspax_input=[], crest_input=[], path_to_workflow=[], descriptor_calculator_input=[], geom='SP'):
        
        self.mace_input = mace_input
        self.chemspax_input = chemspax_input
        self.crest_input = crest_input
        self.path_to_workflow = path_to_workflow
        self.descriptor_calculator_input = descriptor_calculator_input

        self.bidentate_1_index = None
        self.bidentate_2_index = None
        self.metal_index = None
        self.mace_skeletons = dict()
        # Unpack inputs of MACE, Chemspax, CREST
        print('Workflow is initializing. Converting your dict. input to variables.')
        print('')
        self.geom = geom
        self.solvent = None
        if mace_input != []:  
          self.mace_ligands, self.auxiliary_ligands, self.geom, self.central_atom, self.names_of_xyz, self.substrate = self.initialize_mace()
        if chemspax_input != []:
          self.substituent_list, self.path_to_database, self.path_to_substituents = self.initialize_chemspax()
        if crest_input != []:
          self.method, self.charge_of_complex, self.multiplicity, self.solvent, self.conf_search = self.initialize_crest()
        if descriptor_calculator_input != []:
            self.descriptor_central_atom, self.descriptor_metal_adduct, self.descriptor_output_type, self.extract_xyz_from_log, self.descriptor_printout, self.descriptor_solvent = self.initialize_descriptor_calculator()
              
    def initialize_mace(self):
        print('Reading MACE inputs')
        
        bidentate = list(self.mace_input['bidentate_ligands'])
        #bidentate = list(pd.read_excel(self.mace_input['bidentate_ligands'])['smiles'])
        auxiliary_ligands = self.mace_input['auxiliary_ligands']
        geom = self.mace_input['geom']
        central_atom = self.mace_input['central_atom']
        names_of_xyz_key = list(self.mace_input['names_of_xyz'])
        #names_of_xyz_key = list(pd.read_excel(self.mace_input['bidentate_ligands'])['Name'])
        substrate = self.mace_input['substrate']
        return bidentate, auxiliary_ligands, geom, central_atom, names_of_xyz_key, substrate
      
    def initialize_chemspax(self):

        print('Reading ChemSpaX inputs')

        substituent_list = self.chemspax_input['substituent_list']
        # assume that this path is always at the base of the chemspax package, so substituents are read from there
        path_to_substituents = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'install', 'chemspax', 'chemspax', 'substituents_xyz', 'manually_generated')
        path_to_database = os.path.join(path_to_substituents, "central_atom_centroid_database.csv")
        # path_to_database = self.chemspax_input['path_to_database']
        # path_to_substituents = self.chemspax_input['path_to_substituents']
        
        return substituent_list, path_to_database, path_to_substituents
    
    def initialize_crest(self): 
        print('Reading CREST inputs.')
        
        method = self.crest_input['method']
        multiplicity = self.crest_input['multiplicity']
        charge_of_complex = self.crest_input['charge_complex']
        solvent = self.crest_input['solvent']
        conf_search = self.crest_input['conformer_search']
        return method, charge_of_complex, multiplicity, solvent, conf_search

    def initialize_descriptor_calculator(self):
        print('Reading Descriptor Calculator inputs.')
        descriptor_central_atom = self.descriptor_calculator_input['central_atom']
        descriptor_metal_adduct = self.descriptor_calculator_input['metal_adduct']
        descriptor_output_type = self.descriptor_calculator_input['output_type']
        extract_xyz_from_log = self.descriptor_calculator_input['extract_xyz_from_log']
        descriptor_printout = self.descriptor_calculator_input['descriptor_printout']
        descriptor_solvent = self.descriptor_calculator_input['solvent']

        return descriptor_central_atom, descriptor_metal_adduct, descriptor_output_type, extract_xyz_from_log, descriptor_printout, descriptor_solvent

    def prepare_folder_structure(self):
        
        print('Preparing the folder structure')

        if not os.path.exists(self.path_to_workflow):
            os.mkdir(self.path_to_workflow)
        os.chdir(self.path_to_workflow)
        if not os.path.exists('MACE'):
            os.mkdir('MACE')
        if not os.path.exists('ChemSpaX'):
            os.mkdir('ChemSpaX')
        if not os.path.exists('CREST'):
            os.mkdir('CREST')
        if not os.path.exists('Descriptors'):
            os.mkdir('Descriptors') 
        os.chdir(self.path_to_workflow)

    def run_mace(self):
        
        os.chdir(os.path.join(self.path_to_workflow, 'MACE'))

        for i in tqdm(range(len(self.mace_ligands))):
            try:
                if self.geom == 'OH':
                    identifier_OH = self.names_of_xyz[i] + '_OH'
                    complex = MACE(self.mace_ligands[i], self.central_atom,  identifier_OH)
                    complex.generate_complex_OH_xyz(self.auxiliary_ligands, substrate=self.substrate)
                
                if self.geom == 'SP':
                    identifier_SP = self.names_of_xyz[i] + '_SP'
                    complex = MACE(self.mace_ligands[i], self.central_atom, identifier_SP)
                    Xs = complex.generate_complex_SP_xyz(self.auxiliary_ligands)
                    # j index - conformer index
                    # i index - ligand index
                    for j in range(Xs):
                        if self.names_of_xyz[i] in self.mace_skeletons.keys():
                            self.mace_skeletons[self.names_of_xyz[i]].append("_".join([self.central_atom, identifier_SP, str(j) + '.xyz']))
                        else:
                            self.mace_skeletons[self.names_of_xyz[i]] = ["_".join([self.central_atom, identifier_SP, str(j) + '.xyz'])]
            except Exception:
                    print('Wrong SMILES:', i)               

    def run_chemspax(self, names, functionalization_list):  
        
        """
        
        Run chemspax from the pip installable package chemspax.
        Use prepare_data(...) to prepare the substituents.
        Then run main(...) to functionalize the skeletons prepared by MACE.
        
        """
        print('MACE scaffolds: ',self.mace_skeletons)
        self.merged_mace_skeletons = sum(self.mace_skeletons.values(), [])
        
        chemspax_working_directory = os.path.join(self.path_to_workflow, 'ChemSpaX')
        
        # Export MACE skeletons to ChemSpaX
        
        src_dir_skeletons =  os.path.join(self.path_to_workflow, 'MACE')
        
        if self.substituent_list.size == 0:
            dest_dir_skeletons = os.path.join(self.path_to_workflow, 'ChemSpaX', 'chemspax_output')
            shutil.copytree(src_dir_skeletons, dest_dir_skeletons)  

        else:              
            dest_dir_skeletons =  os.path.join(self.path_to_workflow, 'ChemSpaX', 'skeletons')        
            if not os.path.exists(dest_dir_skeletons):      
                shutil.copytree(src_dir_skeletons, dest_dir_skeletons)            
            skeleton_list = glob.glob(os.path.join(self.path_to_workflow , 'ChemSpaX', 'skeletons','*.xyz'))
        
            for xyz_file in skeleton_list:
                change_second_line_xyz(xyz_file, new_content='')

            # set skeleton path to ../ChemSpaX/skeletons        
            path_to_skeletons = os.path.join(os.getcwd(), dest_dir_skeletons)
            # copy substituents from user source to ChemSpaX folder
            # src_dir_subs = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'install', 'chemspax', 'chemspax', 'substituents_xyz', 'manually_generated')
            # src_dir_subs = self.path_to_substituents
            # dest_dir_subs = os.path.join(self.path_to_workflow, 'ChemSpaX', 'substituents_xyz')
            
            # if not os.path.exists(dest_dir_subs):
            #     shutil.copytree(src_dir_subs, dest_dir_subs)

            # Prepare data (from data_preparation.py in the chemspax package)
            print('Data preparation has been performed.')
            # path_to_substituents = src_dir_subs
            #   prepare_data(path_to_substituents, path_to_skeletons, self.path_to_database)

            # Set path to output
            path_to_output = os.path.join(self.path_to_workflow, 'ChemSpaX','chemspax_output')

            # Run chemspax from main

        for index, sub_list in enumerate(self.substituent_list):
            print(self.substituent_list)
            #### Run chemspax

            #### Convert mace_skeletons to full paths (chemspax format).
            chemspax_skeletons = [os.path.join(path_to_skeletons, i) for i in self.mace_skeletons[names[index]]]

            chemspax_main(chemspax_skeletons, sub_list, self.path_to_database, self.path_to_substituents, os.path.join(path_to_skeletons), chemspax_working_directory, path_to_output)

            for skeleton in chemspax_skeletons:
                os.rename(os.path.join(path_to_output, os.path.basename(skeleton)[:-4] + '_func_' + str(len(sub_list)) + '.mol'), \
                    os.path.join(path_to_output, list(names)[index] + '_functionalization_' + str(functionalization_list[index]) + '.mol'))

            ## Convert mol to xyz to keep the bonding information
            obconversion = openbabel.OBConversion()
            obconversion.SetInFormat('mol')
            obconversion.SetOutFormat('xyz')
            mol = openbabel.OBMol()
            obconversion.ReadFile(mol, \
                os.path.join(path_to_output, list(names)[index] + '_functionalization_' + str(functionalization_list[index]) + '.mol'))

            obconversion.WriteFile(mol, \
                os.path.join(path_to_output,list(names)[index] + '_functionalization_' + str(functionalization_list[index]) + '.xyz'))

            # for filename in glob.glob(os.path.join(self.path_to_workflow, 'ChemSpaX', 'chemspax_output', '*_func_*')):
            #     os.remove(filename)

    def run_crest(self, path_to_complexes = [], path_to_output = [], conformer_search = 'off'):
        
        """
        
        This function is designed to either run conformer search with xtb preopt. or just xtb optimization.
        The subprocess.call is used to make a binary call from your ./bashrc file or any local path in your 
        $PATH system variable.
        
        Files are stored in the users preferred place when you run the workflow in a modular way, or by design
        the results are stored in your Workflow/CREST folder.

        """
    
        if path_to_complexes == []:
          path_to_complexes = os.path.join(self.path_to_workflow, 'ChemSpaX', 'chemspax_output')
          os.chdir(self.path_to_workflow)

        
        for path_to_complex in glob.glob(os.path.join(path_to_complexes, '*.xyz')):
            os.chdir(path_to_complexes)
            
            folder_name = path_to_complex[:-4]
            os.mkdir(folder_name)
            os.chdir(folder_name)

            print('About to run {}-xtb optimization:'.format(self.method))

            if conformer_search == 'on':
                subprocess.call('xtb "%s" --%s --chrg %s  --uhf %s --alpb %s --opt > xtb.out' \
                    % (path_to_complex, self.method, self.charge_of_complex, self.multiplicity, self.solvent), shell=True)
                
                print('About to RUN conformer search with CREST:')
                
                subprocess.call('crest xtbopt.xyz --%s --chrg %s  --uhf %s --alpb %s > crest.out' \
                    % (path_to_complex, self.method, self.charge_of_complex, self.multiplicity, self.solvent), shell=True)
       
            else:
                subprocess.call('xtb "%s" --%s --chrg %s  --uhf %s --alpb %s --opt > xtb.out' \
                    % (path_to_complex, self.method, self.charge_of_complex, self.multiplicity, self.solvent), shell=True)
            
            
            src_dir = path_to_complex[:-4]
            
            if path_to_output != []:
                dest_dir = path_to_output
            else:
                dest_dir = os.path.join(self.path_to_workflow, 'CREST')
            
            shutil.move(src_dir, dest_dir) 
            os.chdir(path_to_complexes)
        
        os.chdir(self.path_to_workflow)
            
    def calculate_descriptors(self):
        """
        This function uses the Descriptor class to calculate descriptors for xyz, CREST or log files. The output_type
        is used to determine which descriptors are calculated. The extract_xyz_from_log is used to extract the xyz
        coordinates from the log file. The printout is used to print the descriptors to the screen.

        By default, self.path_to_workflow is used as the path for the files to be read in.
        (so obelix/Workflow)

        :param metal_adduct: pristine, nbd are fully supported, acetonitrile is WIP
        :param output_type: xyz, crest or gaussian
        :param extract_xyz_from_log: True or False (only used for descriptors from gaussian output)
        :param printout: True or False (prints the descriptors to the screen)
        :return: dataframe of descriptors, calculated and loaded in the Descriptor class
        """
        geom_type = self.geom
        # fix the SP and BD confusion (they essentially mean the same thing, the structure is in SP coordination)
        if self.geom == 'SP':
            geom_type = 'BD'

        if self.descriptor_output_type == 'xyz' or self.descriptor_output_type == 'crest':
            descriptor_calculator = Descriptors(central_atom=self.descriptor_central_atom, path_to_workflow=self.path_to_workflow,
                                                output_type=self.descriptor_output_type)

            descriptor_calculator.calculate_morfeus_descriptors(geom_type=geom_type, solvent=self.descriptor_solvent,
                                                                printout=self.descriptor_printout, metal_adduct=self.descriptor_metal_adduct)
            return descriptor_calculator.descriptor_df

        elif self.descriptor_output_type == 'gaussian':
            descriptor_calculator = Descriptors(central_atom=self.descriptor_central_atom, path_to_workflow=self.path_to_workflow,
                                                output_type=self.descriptor_output_type)

            descriptor_calculator.calculate_dft_descriptors_from_log(geom_type=geom_type, solvent=self.descriptor_solvent,
                                                                     extract_xyz_from_log=self.extract_xyz_from_log,
                                                                     printout=self.descriptor_printout, metal_adduct=self.descriptor_metal_adduct)
        else:
            raise ValueError('Output type not recognized')

    def run_workflow(self):
        
        if self.path_to_workflow != []:
           self.prepare_folder_structure()
        
        self.run_mace()
        self.run_chemspax()
        self.run_crest(self.conf_search)
        self.calculate_descriptors()


if __name__ == "__main__":
    
    ligand_excel_file = 'ligands_test.xlsx'
    auxiliary_ligands = []
    substrate = ['CC#[N:1]']

    geom = 'SP'
    central_atom = '[Ir+3]'
    names_of_xyz_key = 'Cas'


    # MACE input 
    mace_input = {'bidentate_ligands': ligand_excel_file, 
                  'auxiliary_ligands': auxiliary_ligands, 
                  'names_of_xyz': names_of_xyz_key, 
                  'central_atom': central_atom, 
                  'geom': geom, 
                  'substrate': substrate}

    # ChemXpaX input
    
    # current_directory = os.path.join(os.getcwd(), "chemspax")  # use path.join()
    # # os.chdir(current_directory)
    # path_to_substituents = os.path.join(current_directory, "substituents_xyz") 
    # path_to_database = os.path.join(path_to_substituents, "central_atom_centroid_database.csv")
    # substituent_df = pd.read_excel('new_dataset.xlsx').dropna()
    # substituent_list = np.array(substituent_df[['R1', 'R2', 'R3', 'R4']])
    # # print(substituent_list)
    # names = substituent_df['Name']

    # skeleton_list = list(substituent_df['Skeleton'])
    # # print(skeleton_list)
    # path_to_hand_drawn_skeletons = os.path.join(current_directory, "skeletons")
    # path_to_output = os.path.join(current_directory, "complexes")

    # chemspax_input = {'skeleton_list' : skeleton_list, 
    #                 'substituent_list' : substituent_list, 
    #                 'path_to_database' : path_to_database, 
    #                 'path_to_substituents' : path_to_substituents, 
    #                 'path_to_additional_skeletons' : path_to_hand_drawn_skeletons}


    method = 'gfn2'
    charge_of_complex = 0
    multiplicity = 2
    solvent = 'ch2cl2'
    
    crest_input = {'method': method, 
                   'charge_complex': charge_of_complex,
                   'multiplicity' : multiplicity,
                   'solvent' : solvent,
                   'conformer_search' : 'off'}

    descriptor_central_atom = 'Rh'
    descriptor_metal_adduct = 'pristine'
    descriptor_output_type = 'xyz'
    extract_xyz_from_log = True
    descriptor_printout = False

    descriptor_input = {'central_atom': descriptor_central_atom,
                        'metal_adduct': descriptor_metal_adduct,
                        'output_type': descriptor_output_type,
                        'extract_xyz_from_log': extract_xyz_from_log,
                        'descriptor_printout': descriptor_printout,
                        'solvent': solvent}

    # print(skeleton_list)
    workflow = Workflow(path_to_workflow = os.getcwd() + '/Workflow', geom='SP', descriptor_calculator_input=descriptor_input)
    workflow.prepare_folder_structure()
    # descriptor_df = workflow.calculate_descriptors()
    # descriptor_df.to_csv('descriptor_df_test.csv', index=False)
    # workflow.run_chemspax(names=names ,functionalization_list=skeleton_list)
    
