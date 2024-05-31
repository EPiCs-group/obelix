"""
Test whether OBeLiX produces the expected descriptor values for a given set of xyz files generated by MACE.

"""

import os
import shutil
from typing import Any

import numpy as np
import pandas as pd
import pytest
import logging


# from obelix.descriptor_calculator import Descriptors
# from obelix.run_workflow import Workflow

import obelix.descriptor_calculator
import obelix.run_workflow

# path to the folder where the output of MACE (xyz files) and OBeLix (descriptors.csv) will be stored. This folder gets generated as a result of the MACE workflow being called.
path_to_workflow = os.path.abspath(os.path.join(os.path.dirname(__file__), "output"))


mace_input = [
    {
        "bidentate_ligands": [
            "c1ccc([P:1](CC[P:1](c2ccccc2)c2cccc3ccccc23)c2cccc3ccccc23)cc1"
        ],
        "auxiliary_ligands": ["CC#[N:1]", "CC#[N:1]"],
        "names_of_xyz": ["1-Naphthyl-DIPAMP"],
        "central_atom": "[Rh+]",
        "geom": "SP",
        "substrate": [],
    },
]

expected_csv_files = [
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "expected_output",
            "Descriptors",
            "descriptors_mace_input_0.csv",
        )
    ),
]


mace_input_and_expected_csv = list(zip(mace_input, expected_csv_files))


@pytest.fixture(params=mace_input_and_expected_csv)
def mace_input_and_expected_csv_fix(request):
    return request.param


@pytest.fixture
def xyz_files(mace_input_and_expected_csv_fix: Any):
    """
    Run the MACE workflow to generate the xyz files for the input defined in the fixture input_for_mace.

    Ouput of MACE is a set of xyz files and will be stored in the 'output/' folder in the current working directory.
    """
    workflow = obelix.run_workflow.Workflow(
        mace_input=mace_input_and_expected_csv_fix[0],
        path_to_workflow=path_to_workflow,
        geom="BD",
    )
    logging.info("object id of workflow: %s", id(workflow))
    # create the folder structure
    workflow.prepare_folder_structure()

    # create and write xyz files for the complexes defined by the input
    workflow.run_mace()
    xyz_files = os.path.join(path_to_workflow, "MACE")
    yield xyz_files
    # clean up the 'output/' folder
    shutil.rmtree(path_to_workflow)


@pytest.fixture
def output_csv(xyz_files: str):
    """
    Calculate the descriptors for the xyz structures and write the descriptors to a csv file.
    """
    # path to MACE folder
    path_to_mace_output = xyz_files

    # calculate the descriptors for the xyz structures
    descriptors = obelix.descriptor_calculator.Descriptors(
        central_atom="Rh",
        path_to_workflow=path_to_mace_output,
        output_type="xyz",
    )
    logging.info("object id of descriptors: %s", id(descriptors))
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


@pytest.fixture
def output_df(output_csv: str):
    """Return the output csv file as a pandas dataframe."""
    # check if the output csv file exists and then read it as a pandas dataframe
    if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
        output_df = pd.read_csv(output_csv)
    else:
        raise ValueError(f"File {output_csv} is empty or does not exist.")
    return output_df


@pytest.fixture
def expected_csv(mace_input_and_expected_csv_fix: Any):
    return mace_input_and_expected_csv_fix[1]


@pytest.fixture
def expected_df(expected_csv: str) -> pd.DataFrame:
    """Return the expected output csv file as a pandas dataframe."""
    # check if the expected output csv file exists and then read it as a pandas dataframe
    if os.path.exists(expected_csv) and os.path.getsize(expected_csv) > 0:
        expected_df = pd.read_csv(expected_csv)
    else:
        raise ValueError(f"File {expected_csv} is empty or does not exist.")
    return expected_df


########################### Testcases ###########################


def test_descriptor_values(output_df: pd.DataFrame, expected_df: pd.DataFrame):

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


# def test_descriptor_values(output_df: pd.DataFrame, expected_output_df: pd.DataFrame):

#     # Read the expected output csv file as a pandas dataframe

#     # Filter the columns containing the descriptor values
#     output_descriptor_values_df = output_df.loc[
#         :, ~output_df.columns.str.contains("index|element|filename_tud")
#     ]

#     expected_descriptor_values_df = expected_output_df.loc[
#         :, ~expected_output_df.columns.str.contains("index|element|filename_tud")
#     ]

#     # Convert the dataframes to numpy arrays for comparison
#     output_descriptor_values = output_descriptor_values_df.to_numpy()
#     expected_descriptor_values = expected_descriptor_values_df.to_numpy()
#     assert np.allclose(
#         output_descriptor_values, expected_descriptor_values
#     ), "The descriptor values in the output csv file does not match the expected descriptor values for this input."


# def test_index_values(output_df: pd.DataFrame, expected_df: pd.DataFrame):
#     output_index_values_df = output_df.loc[:, output_df.columns.str.contains("index")]
#     expected_index_values_df = expected_df.loc[
#         :, expected_df.columns.str.contains("index")
#     ]
#     assert output_index_values_df.equals(
#         expected_index_values_df
#     ), "The index values in the output csv file does not match the expected index values for this input."


# def test_element_values(output_df: pd.DataFrame, expected_df: pd.DataFrame):
#     output_element_values_df = output_df.loc[
#         :, output_df.columns.str.contains("element")
#     ]
#     expected_element_values_df = expected_df.loc[
#         :, expected_df.columns.str.contains("element")
#     ]
#     assert output_element_values_df.equals(
#         expected_element_values_df
#     ), "The element values in the output csv file does not match the expected element values for this input."


# def test_filename_values(output_df: pd.DataFrame, expected_df: pd.DataFrame):
#     assert output_df["filename_tud"].equals(
#         expected_df["filename_tud"]
#     ), "The filename values in the output csv file does not match the expected filename values for this input."


#################### Clean up ####################


# @pytest.fixture(scope="module", autouse=True)
# def clean_up():
#     """
#     Delete the 'output/' folder generated as a result of the MACE workflow.
#     """
#     yield
#     shutil.rmtree(path_to_workflow)
