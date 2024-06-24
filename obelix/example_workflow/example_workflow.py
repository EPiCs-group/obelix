from obelix.run_workflow import *

ligand_excel_file = os.path.join(os.getcwd(), 'test_mace.xlsx')
ligand_df = pd.read_excel(ligand_excel_file).dropna()
ligand_name = ligand_df['Name']
ligand_smiles = ligand_df['smiles']

auxiliary_ligands = ['CC#[N:1]', 'CC#[N:1]']
substrate = ['CC#[N:1]']

geom = 'SP'
central_atom = '[Rh+]'



# MACE input 
mace_input = {'bidentate_ligands': ligand_smiles, 
                'auxiliary_ligands': auxiliary_ligands, 
                'names_of_xyz': ligand_name, 
                'central_atom': central_atom, 
                'geom': geom, 
                'substrate': substrate}

chemspax_directory = os.path.join('install', "chemspax", 'chemspax')  # use path.join()
# os.chdir(current_directory)
path_to_substituents = os.path.join(chemspax_directory, "substituents_xyz")
path_to_database = os.path.join(path_to_substituents, "manually_generated", "central_atom_centroid_database.csv")
substituent_df = pd.read_excel(os.path.join(os.getcwd(), 'test_chemspax.xlsx'), sheet_name='Sheet1').dropna()
substituent_list = np.array(substituent_df[['R1', 'R2', 'R3', 'R4']])
# print(substituent_list)
names = substituent_df['Name']
func_list = substituent_df['Functionalization']
# print(skeleton_list)
# path_to_hand_drawn_skeletons = os.path.join(os.getcwd(), "skeletons")
# path_to_output = os.path.join(os.getcwd(), "complexes")

chemspax_input = {'substituent_list' : substituent_list}
                # 'path_to_database' : path_to_database,
                # 'path_to_substituents' : path_to_substituents}

workflow = Workflow(mace_input = mace_input, chemspax_input=chemspax_input, path_to_workflow = os.getcwd() + '/wf_test5', geom='BD')
workflow.prepare_folder_structure()
workflow.run_mace()
workflow.run_chemspax(names=names, functionalization_list=func_list)
