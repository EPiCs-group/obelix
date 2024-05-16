"""
Test whether OBeLiX produces the expected descriptor values for a given set of xyz files generated by MACE.

"""

import os
import shutil
from typing import Any

import numpy as np
import pandas as pd
import pytest

from obelix.descriptor_calculator import Descriptors
from obelix.run_workflow import Workflow

# path to the folder where the output of MACE (xyz files) and OBeLix (descriptors.csv) will be stored. This folder gets generated as a result of the MACE workflow being called.
path_to_workflow = os.path.abspath(os.path.join(os.path.dirname(__file__), "output"))

####################### Input for MACE ######################


# To do: specify input as a dictionary, then parameterize the fixture. Generate descriptor.csv for each input and keep in the expected_output/Dercriptors folder with the name descript_mace_config_1.csv, descript_mace_config_2.csv, etc. Check if mace_config is a suitable name for the mace input we define below. Since we do not compare the xyz files, we do not need to store them in the expected_output folder.
@pytest.fixture(scope="module")
def input_for_mace():
    """
    Define a square planar geometry with a central Rhodium atom as the metal center. Square planar means that the metal center has 4 ligands in the same plane the other option is 'OH' for octahedral geometry.
    """
    geom = "SP"
    central_atom = "[Rh+]"

    # SMILES and name of the ligand with two points of attachment to the metal center
    ligand_name = ["1-Naphthyl-DIPAMP"]
    ligand_smiles = ["c1ccc([P:1](CC[P:1](c2ccccc2)c2cccc3ccccc23)c2cccc3ccccc23)cc1"]

    # SMILES of the auxiliary ligands with one point of attachment to the metal center
    auxiliary_ligands = ["CC#[N:1]", "CC#[N:1]"]

    # in the SP geometry, the ligands are in the same plane, so we don't need a substrate for this example
    substrate = []

    # the input for MACE is a dictionary with the following keys
    mace_input = {
        "bidentate_ligands": ligand_smiles,
        "auxiliary_ligands": auxiliary_ligands,
        "names_of_xyz": ligand_name,
        "central_atom": central_atom,
        "geom": geom,
        "substrate": substrate,
    }

    return mace_input


################ Input for Descriptor calculation ##############


@pytest.fixture(scope="module")
def xyz_files(input_for_mace: dict[str, Any]):
    """
    Run the MACE workflow to generate the xyz files for the input defined in the fixture input_for_mace.

    Ouput of MACE is a set of xyz files and will be stored in the 'output/' folder in the current working directory.
    """
    workflow = Workflow(
        mace_input=input_for_mace,
        path_to_workflow=path_to_workflow,
        geom="BD",
    )

    # create the folder structure
    workflow.prepare_folder_structure()

    # create and write xyz files for the complexes defined by the input
    workflow.run_mace()
    xyz_files = os.path.join(path_to_workflow, "MACE")
    return xyz_files


########################### Descriptor calculation ###########################


@pytest.fixture(scope="module")
def output_csv(xyz_files: str):
    """
    Calculate the descriptors for the xyz structures and write the descriptors to a csv file.
    """
    # path to MACE folder
    path_to_mace_output = xyz_files

    # calculate the descriptors for the xyz structures
    descriptors = Descriptors(
        central_atom="Rh",
        path_to_workflow=path_to_mace_output,
        output_type="xyz",
    )
    descriptors.calculate_morfeus_descriptors(
        geom_type="BD", solvent=None, printout=False, metal_adduct="pristine"
    )

    # write the descriptors to a csv file in the Descriptors folder
    descriptors.descriptor_df.to_csv(
        os.path.join(
            path_to_workflow,
            "Descriptors",
            "descriptors.csv",
        ),
        index=False,
    )
    output_csv = os.path.join(
        path_to_workflow,
        "Descriptors",
        "descriptors.csv",
    )
    return output_csv


@pytest.fixture(scope="module")
def output_df(output_csv: str):
    """Return the output csv file as a pandas dataframe."""
    output_df = pd.read_csv(output_csv)
    return output_df


@pytest.fixture(scope="module")
def expected_csv():
    path_to_expected_csv = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "expected_output",
            "Descriptors",
        )
    )
    expected_csv = os.path.join(path_to_expected_csv, "descriptors.csv")
    return expected_csv


@pytest.fixture(scope="module")
def expected_df(expected_csv: str):
    """Return the expected csv file as a pandas dataframe."""
    expected_df = pd.read_csv(expected_csv)
    return expected_df


########################### Testcases ###########################


def test_descriptor_values(output_df: pd.DataFrame, expected_df: pd.DataFrame):

    # Filter the columns containing the descriptor values
    output_descriptor_values_df = output_df.loc[
        :, ~output_df.columns.str.contains("index|element|filename")
    ]
    expected_descriptor_values_df = expected_df.loc[
        :, ~expected_df.columns.str.contains("index|element|filename_tud")
    ]

    # Convert the dataframes to numpy arrays for comparison
    output_descriptor_values = output_descriptor_values_df.to_numpy()
    expected_descriptor_values = expected_descriptor_values_df.to_numpy()
    assert np.allclose(
        output_descriptor_values, expected_descriptor_values
    ), "The descriptor values in the output csv file does not match the expected descriptor values for this input."


def test_index_values(output_df: pd.DataFrame, expected_df: pd.DataFrame):
    output_index_values_df = output_df.loc[:, output_df.columns.str.contains("index")]
    expected_index_values_df = expected_df.loc[
        :, expected_df.columns.str.contains("index")
    ]
    assert output_index_values_df.equals(
        expected_index_values_df
    ), "The index values in the output csv file does not match the expected index values for this input."


def test_element_values(output_df: pd.DataFrame, expected_df: pd.DataFrame):
    output_element_values_df = output_df.loc[
        :, output_df.columns.str.contains("element")
    ]
    expected_element_values_df = expected_df.loc[
        :, expected_df.columns.str.contains("element")
    ]
    assert output_element_values_df.equals(
        expected_element_values_df
    ), "The element values in the output csv file does not match the expected element values for this input."


def test_filename_values(output_df: pd.DataFrame, expected_df: pd.DataFrame):
    assert output_df["filename_tud"].equals(
        expected_df["filename_tud"]
    ), "The filename values in the output csv file does not match the expected filename values for this input."


#################### Clean up ####################


@pytest.fixture(scope="module", autouse=True)
def clean_up():
    """
    Delete the 'output/' folder generated as a result of the MACE workflow.
    """
    yield
    shutil.rmtree(path_to_workflow)
