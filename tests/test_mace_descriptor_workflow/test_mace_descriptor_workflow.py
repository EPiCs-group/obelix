"""
Test whether OBeLiX produces the expected descriptor values for a given set of xyz files generated by MACE.

*************** Test design ***************

Three testcases are defined in this module:
1. test_descriptor_values: Compare the descriptor values in the output csv file with the expected descriptor values for the input defined in the fixture input_for_mace.
2. test_index_values: Compare the index values in the output csv file with the expected index values for the input defined in the fixture input_for_mace.
3. test_element_values: Compare the element values in the output csv file with the expected element values for the input defined in the fixture input_for_mace.

The data that each of the above testcases need are provided by fixtures passed as arguments to the testcases. Each fixture is a function that returns the data needed by the testcases.

Since the setup required for each testcase consistes of several steps that execute sequentially, the fixtures are chained together. The output of one fixture is passed as an argument to the next fixture.

scope='module' is used for the fixtures to ensure that the fixtures are run only once for all the testcases defined in this module. This is because the fixtures are computationally expensive and the data generated by the fixtures are not modified by the testcases.

*************** Running the tests ***************

The tests in this module can be run using the following command:
`pytest tests/test_mace_descriptor_workflow/test_mace_descriptor_workflow.py` or by simply running `pytest` in the terminal from the root directory of the repository. This command will run all the testcases defined in this module in the order they are defined.

When the tests are run, the following sequence of events occur:

1. The input_for_mace fixture is run to generate the input for MACE.
2. The xyz_files fixture is run to generate the xyz files for the input defined in the input_for_mace fixture.
3. The output_csv fixture is run to calculate the descriptors for the xyz files generated by MACE and write the descriptors to a csv file.
4. The output_df fixture is run to read the output csv file as a pandas dataframe.
5. The expected_csv fixture is run to get the path to the expected csv file.
6. The expected_df fixture is run to read the expected csv file as a pandas dataframe.
7. The testcases are run with the data provided by the fixtures.

The clean_up fixture is run after all the testcases are executed to delete the 'output/' folder generated as a result of the MACE workflow.

"""

import os
import shutil
from typing import Any

import numpy as np
import pandas as pd
import pytest

from obelix.descriptor_calculator import Descriptors
from obelix.run_workflow import Workflow

# 'output/' holds the outputs generated by the MACE workflow and the descriptor calculator triggered as part of the test. The output includes the MACE folder containing the xyz files generated by MACE and the Descriptors folder containing the descriptors.csv file generated by the descriptor calculator.
path_workflow_output = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "output")
)


@pytest.fixture(scope="module")
def input_for_mace():
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

    Parameters
    ----------
    input_for_mace : dict[str, Any]
        The input for MACE. The input is provided by the fixture input_for_mace.

    Returns
    -------
    str
        The path to the folder containing the xyz files generated by MACE.

    """
    workflow = Workflow(
        mace_input=input_for_mace,
        path_to_workflow=path_workflow_output,
        geom="BD",
    )

    # create the folder structure
    workflow.prepare_folder_structure()

    # create and write xyz files for the complexes defined by the input
    workflow.run_mace()
    xyz_files = os.path.join(path_workflow_output, "MACE")
    return xyz_files


########################### Descriptor calculation ###########################


@pytest.fixture(scope="module")
def output_csv(xyz_files: str):
    """
    Calculate the descriptors for the xyz structures and write the descriptors to a csv file.

    Parameters
    ----------
    xyz_files : str
        The path to the folder containing the xyz files generated by MACE. The path is provided by the fixture xyz_files.

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
            path_workflow_output,
            "Descriptors",
            "descriptors.csv",
        ),
        index=False,
    )
    output_csv = os.path.join(
        path_workflow_output,
        "Descriptors",
        "descriptors.csv",
    )
    return output_csv


@pytest.fixture(scope="module")
def output_df(output_csv: str):
    """
    Return the output csv file as a pandas dataframe.

    Parameters
    ----------
    output_csv : str
        The path to the output csv file prodiced as part of the descriptor calculation. The path is provided by the fixture output_csv.

    """
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
    """
    Compare the descriptor values in the output csv file with the expected descriptor values for the input defined in the fixture input_for_mace.

    Parameters
    ----------
    output_df : pd.DataFrame
        The output csv file as a pandas dataframe. The dataframe is provided by the fixture output_df.
    expected_df : pd.DataFrame
        The expected csv file as a pandas dataframe. The dataframe is provided by the fixture expected_df.

    """

    # Filter the columns containing the descriptor values
    output_descriptor_values_df = output_df.loc[
        :, ~output_df.columns.str.contains("index|element|filename_tud")
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
    shutil.rmtree(path_workflow_output)
