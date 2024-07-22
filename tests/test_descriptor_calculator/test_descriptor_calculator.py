import os
import pandas as pd
import pytest
import numpy as np
import shutil
from obelix.descriptor_calculator import Descriptors


@pytest.fixture
def setup_descriptors(request):
    scenario = request.param
    path_to_xyz_files = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "data", "xyz_files", scenario
    )
    descriptors = Descriptors(
        central_atom="Rh",
        path_to_workflow=path_to_xyz_files,
        output_type="xyz",
    )
    descriptors.calculate_morfeus_descriptors(
        geom_type="BD", solvent=None, printout=False, metal_adduct=scenario
    )
    output_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Store the output csv file in the output folder
    output_file_path = os.path.join(output_dir, "descriptors.csv")
    descriptors.descriptor_df.to_csv(output_file_path, index=False)

    yield descriptors, scenario

    # Cleanup: Delete the output folder and all its contents
    shutil.rmtree(output_dir)


@pytest.mark.parametrize("setup_descriptors", ["nbd", "pristine"], indirect=True)
class TestDescriptorCalculation:
    def test_descriptor_values(self, setup_descriptors):
        # path to expected csv is descriptors_nbd.csv or descriptors_pristine.csv
        descriptors, scenario = setup_descriptors
        path_expected_csv = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "expected_output",
            f"descriptors_{scenario}.csv",
        )
        output_csv = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "output",
            "descriptors.csv",
        )

        expected_df = pd.read_csv(path_expected_csv)
        output_df = pd.read_csv(output_csv)

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

    def test_index_values(self, setup_descriptors):
        # path to expected csv is descriptors_nbd.csv or descriptors_pristine.csv
        descriptors, scenario = setup_descriptors
        path_expected_csv = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "expected_output",
            f"descriptors_{scenario}.csv",
        )
        output_csv = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "output",
            "descriptors.csv",
        )

        expected_df = pd.read_csv(path_expected_csv)
        output_df = pd.read_csv(output_csv)

        output_index_values_df = output_df.loc[
            :, output_df.columns.str.contains("index")
        ]

        expected_index_values_df = expected_df.loc[
            :, expected_df.columns.str.contains("index")
        ]
        assert output_index_values_df.equals(
            expected_index_values_df
        ), "The index values in the output csv file does not match the expected index values for this input."

    def test_element_values(self, setup_descriptors):
        # path to expected csv is descriptors_nbd.csv or descriptors_pristine.csv
        descriptors, scenario = setup_descriptors
        path_expected_csv = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "expected_output",
            f"descriptors_{scenario}.csv",
        )
        output_csv = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "output",
            "descriptors.csv",
        )

        expected_df = pd.read_csv(path_expected_csv)
        output_df = pd.read_csv(output_csv)

        output_element_values_df = output_df.loc[
            :, output_df.columns.str.contains("element")
        ]
        expected_element_values_df = expected_df.loc[
            :, expected_df.columns.str.contains("element")
        ]
        assert output_element_values_df.equals(
            expected_element_values_df
        ), "The element values in the output csv file does not match the expected element values for this input."

    def test_filename_values(self, setup_descriptors):
        # path to expected csv is descriptors_nbd.csv or descriptors_pristine.csv
        descriptors, scenario = setup_descriptors
        path_expected_csv = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "expected_output",
            f"descriptors_{scenario}.csv",
        )
        output_csv = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "output",
            "descriptors.csv",
        )

        expected_df = pd.read_csv(path_expected_csv)
        output_df = pd.read_csv(output_csv)

        assert output_df["filename_tud"].equals(
            expected_df["filename_tud"]
        ), "The filename values in the output csv file does not match the expected filename values for this input."
