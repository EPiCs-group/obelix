# OBeLiX
![logo](../images/logo.png)
Open Bidentate Ligand eXplorer (OBeLiX) is a workflow for automated and reproducible TM-based catalyst structure generation and descriptor calculation. It Uses
our in-house developed packages [MACE](https://github.com/EPiCs-group/epic-mace) and [ChemSpaX](https://github.com/epics-group/chemspax) for
bias-free catalyst structure generation. [CREST](https://github.com/crest-lab/crest) is used for conformer search.
Automated descriptor extraction/calculation is done using [Morfeus](https://github.com/digital-chemistry-laboratory/morfeus) and [cclib](https://github.com/cclib/cclib). The workflow is currently usable for metal-ligand complexes of monodentate and bidentate ligands.

# Table of Contents
1. [Installation](#installation)
2. [Folder structure](#folder-structure)
3. [Functionalities](#functionalities)
4. [Workflow structure](#workflow-structure)
5. [Citation](#citation)

## Installation
After cloning the repository, the conda environment with dependencies can be created from the environment.yml file with conda:

```bash
git clone https://github.com/EPiCs-group/obelix
cd obelix
conda env create -f environment.yml
```
The obelix package can then be installed with pip from setup.py in the obelix folder:

```bash
pip install -e .
```

## Folder structure
For each step of the workflow, the results are stored in a separate and predefined folder structure which is as follows:

```bash
Workflow/MACE
Workflow/ChemSpaX
Workflow/CREST
Workflow/Descriptors
```

The output folder structure is a crucial first step in using the workflow, it is recommended to use the following commands to create the folder structure:

```python
from obelix.run_workflow import Workflow # import the workflow class if this you are not running code in run_workflow.py

workflow = Workflow(path_to_workflow = "your/path/to/workflow_results")
workflow.prepare_folder_structure()
```
This will create a 'Workflow' folder at the indicated path with the necessary subdirectories. 


## Functionalities
As mentioned, the workflow consists of four modules, each with their own functionality. The modules can be used separately or all at once.
First the modules will be explained, then the workflow structrure and examples will be explained.
### Structure generation (MACE)
MACE is used to discover configurations for square planar and octahedral metal-ligand complexes. The generated 3D coordinates are optimized at FF level.
The ligands and the metal center are given as SMILES and the 'ComplexFromLigands' function from MACE is used to generate the 3D coordinates and possible ligand-metal configurations.
Afterwards the configuration with the lowest energy is selected and written to an xyz file.
The main functionalities are contained in run_workflow.py.
### Chemical space exploration (ChemSpaX)
ChemSpaX is used to create variations of a chemical structure by substituting dummy atoms on a scaffold (skeleton structure) with substituents (functional groups) from a database.
The newly placed functional groups are optimized with FF and the resulting structures are stored in a folder. 
The functional groups to place are given as a string input (name of the group), but in the example below this is contained in an excel file.
The main functionalities are contained in run_workflow.py and a customized version of ChemSpaX in install/ChemSpaX.
### Conformer search (CREST)
Conformer search is done using CREST. To use this functionality, the user needs to have the CREST binaries downloaded and added to the system's PATH variable.
An xTB optimization is done prior to the conformer search. By default the conformer search is performed on all xyz's in the ChemSpaX output folder unless another folder is specified.
The main functionalities are contained in run_workflow.py.
### Descriptor calculation (Morfeus/cclib)
**Note: The descriptor calculator is currently only usable on Linux or Mac due to a dependency on xtb-python.**
The descriptor calculator is an automated method for calculating descriptors from either xyz files, Gaussian log files or CREST output folders. 
The descriptors are calculated using Morfeus and cclib. The descriptors are calculated for all files in the specified folder.
The molecular_graph method will search for donor atoms (P,N or S) around the specified metal center (central atom) and assign numbering (min or max) based on the donor's xTB partial charge.
This numbering is then used to calculate steric, electronic and geometric descriptors.
The main functionalities are contained in dft_extraction.py (cclib), descriptor_calculator.py (Morfeus/cclib) and run_workflow.py.


## Workflow Structure
### Using all modules at once
You can run the whole workflow at once, like this:

```python
workflow = Workflow(mace_input = mace_input, chemspax_input = chemspax_input, crest_input = crest_input, path_to_workflow = "your/path/to/workflow_results")
workflow.run_workflow()
```

The input to the workflow class contains the input to MACE, input to ChemSpax, input to xTB/CREST and the descriptor calculator. The inputs should be given as below (the run_workflow.py file contains examples as well):

```python
from obelix.run_workflow import Workflow
import os
import pandas as pd
import numpy as np

# preparation of input for MACE
ligand_excel_file = os.path.join(os.getcwd(), 'test_mace.xlsx')  # excel file with name of ligands for screening and their SMILES
auxiliary_ligands = ['CC#[N:1]', 'CC#[N:1]']  # list of SMILES of auxiliary ligands (stay constant for all structures)
substrate = ['CC#[N:1]'] ## for octahedral (OH) complexes it is possible to add a substrate (e.g. acetonitrile)

geom = 'OH'  # 'SP' or 'OH' for square planar or octahedral complexes
central_atom = '[Rh+]'  # SMILES of central atom
names_of_xyz_key = 'Cas'  # column name of ligand names in excel file


# MACE input 
mace_input = {'bidentate_ligands': ligand_excel_file, 
                'auxiliary_ligands': auxiliary_ligands, 
                'names_of_xyz': names_of_xyz_key, 
                'central_atom': central_atom, 
                'geom': geom, 
                'substrate': substrate}

# preparation of input for ChemSpax
current_directory = os.path.join(os.getcwd(), "chemspax")  # navigate to the installed chemspax folder
path_to_substituents = os.path.join(current_directory, "substituents_xyz") # path to folder with substituents
path_to_database = os.path.join(path_to_substituents, "central_atom_centroid_database.csv") # path to database with substituents names

substituent_df = pd.read_excel('test_chemspax.xlsx').dropna()  # excel file with substituents to place on skeleton (column names are indicated sites)
substituent_list = np.array(substituent_df[['R1', 'R2', 'R3', 'R4']])  # list of substituents that will be placed on either R1, R2, R3 or R4 for each skeleton
names = substituent_df['Name'] # names of skeletons (scaffolds) on which substituents will be placed
skeleton_list = substituent_df['Functionalization'] # number of the skeleton 
path_to_hand_drawn_skeletons = os.path.join(current_directory, "skeletons")
path_to_output = os.path.join(current_directory, "complexes")

# ChemSpax input
chemspax_input = {'substituent_list' : substituent_list, 
                'path_to_database' : path_to_database, 
                'path_to_substituents' : path_to_substituents}

workflow = Workflow(mace_input = mace_input, chemspax_input=chemspax_input, path_to_workflow = os.getcwd() + '/wf_test5', geom='BD')
workflow.prepare_folder_structure()
workflow.run_mace()
workflow.run_chemspax(names=names, functionalization_list=skeleton_list)

```

### Using the modules separately
An example of doing only descriptor calculation:

```python
from obelix.run_workflow import Workflow
import os

descriptor_central_atom = 'Rh' # metal center of the complex
descriptor_metal_adduct = 'pristine'  # adduct connected to metal center, either 'pristine' or 'nbd'
# extra descriptors are available if the adduct is 'nbd' (norbornadiene)
descriptor_output_type = 'xyz' # output type of the descriptor calculator, either 'xyz', 'gaussian' or 'crest'
extract_xyz_from_log = False # if True, the xyz coordinates will be extracted from the Gaussian log file 
descriptor_printout = False # if True, the descriptor calculator will print out the descriptors for each structure
solvent = 'ch2cl2' # solvent used in the calculation, see available solvents in xTB documentation

descriptor_input = {'central_atom': descriptor_central_atom,
                    'metal_adduct': descriptor_metal_adduct,
                    'output_type': descriptor_output_type,
                    'extract_xyz_from_log': extract_xyz_from_log,
                    'descriptor_printout': descriptor_printout,
                    'solvent': solvent}

workflow = Workflow(path_to_workflow = os.getcwd() + '/Workflow', geom='SP', descriptor_calculator_input=descriptor_input)
workflow.prepare_folder_structure()
descriptor_df = workflow.calculate_descriptors()
descriptor_df.to_csv('descriptor_df_test.csv', index=False)
 ```

## Citation
WIP