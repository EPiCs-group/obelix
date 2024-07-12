# -*- coding: utf-8 -*-
#                                                     #
#  __author__ = Adrian Mirza & Adarsh Kalikadien      #
#  __institution__ = TU Delft                         #
#  __contact__ = a.v.kalikadien@tudelft.nl            #
import sys, os, glob

import morfeus
from morfeus import read_xyz, BiteAngle, ConeAngle, BuriedVolume, Dispersion, SASA
from morfeus.conformer import ConformerEnsemble
from morfeus.io import read_cclib, write_xyz
from morfeus.utils import convert_elements
import xtb.utils  # comment this if working on windows
import numpy as np
import pandas as pd
from tqdm import tqdm

from obelix.molecular_graph import molecular_graph
from obelix.tools.utilities import dataframe_from_dictionary, calculate_distance, calculate_dihedral
from obelix.dft_extraction import DFTExtractor, NBDComplex


class Descriptors:
    """

    Class for calculating Morfeus and DFT descriptors. Works for Gaussian log files, CREST folders and xyz files.

    """
    def __init__(self, central_atom, path_to_workflow = None, output_type = None, ce = None):
        self.central_atom = central_atom
        self.path_to_workflow = path_to_workflow
        self.supported_output_types = ['xyz', 'crest', 'gaussian']
        #if output_type not in self.supported_output_types:
            #raise ValueError(f'Output type {output_type} not supported. Please choose from {self.supported_output_types}.')
        self.output_type = output_type
        self.descriptor_df = None
        self.ce = ce

    @staticmethod
    def _find_bidentate_ligand(elements, coordinates):
        """
        Graph-based approach used to find the bidentate ligand in a complex. This method returns all auxillary ligand
        atoms and the bidentate ligand atoms. For now we only use the bidentate ligand atoms.

        :param elements:
        :param coordinates:
        :param geom_type:
        :return:
        """
        ligand_atoms, bidentate = molecular_graph(elements=elements, coordinates=coordinates)
        return ligand_atoms, bidentate

    @ staticmethod
    def _calculate_c_c_distance_nbd(elements, coordinates, dictionary):
        """
        Calculate the distance between the double bonds that pi coordinate to the metal in a NBD geometry.
        In this case the NBD geometry was predefined and the indices are known. (always at the bottom of the file)

        :param elements:
        :param coordinates:
        :param dictionary:
        :return:
        """
        elements = convert_elements(elements, 'symbols')
        # indexing for all nbd structures is the same, so we can use the same indices for all
        coordinates_c1 = coordinates[-6]
        coordinates_c2 = coordinates[-9]
        distance_pi_bond_1 = calculate_distance(coordinates_c1, coordinates_c2)
        dictionary["distance_pi_bond_1"] = distance_pi_bond_1
        dictionary["distance_pi_bond_1_element_1"] = elements[-6]
        dictionary["distance_pi_bond_1_element_2"] = elements[-9]
        dictionary["distance_pi_bond_1_element_1_idx"] = np.where(np.array(coordinates) == coordinates_c1)[0][0] + 1
        dictionary["distance_pi_bond_1_element_2_idx"] = np.where(np.array(coordinates) == coordinates_c2)[0][0] + 1

        coordinates_c3 = coordinates[-7]
        coordinates_c4 = coordinates[-10]
        distance_pi_bond_2 = calculate_distance(coordinates_c3, coordinates_c4)
        dictionary["distance_pi_bond_2"] = distance_pi_bond_2
        dictionary["distance_pi_bond_2_element_1"] = elements[-7]
        dictionary["distance_pi_bond_2_element_2"] = elements[-10]
        dictionary["distance_pi_bond_2_element_1_idx"] = np.where(np.array(coordinates) == coordinates_c3)[0][0] + 1
        dictionary["distance_pi_bond_2_element_2_idx"] = np.where(np.array(coordinates) == coordinates_c4)[0][0] + 1

        return dictionary

    @ staticmethod
    def _calculate_dihedral_angles_nbd_and_metal_donors(dictionary, metal_idx, bidentate_max_donor_idx, bidentate_min_donor_idx, elements, coordinates, central_carbon_nbd_idx, hydrogens_bonded_to_carbon_back_nbd_idxs):
        """
        Calculate the H-C-M-P dihedral angles for the NBD geometry. In this case the NBD geometry was predefined and the
        indices are known. (always at the bottom of the file)

        :param dictionary:
        :param metal_idx:
        :param bidentate_max_donor_idx:
        :param bidentate_min_donor_idx:
        :param elements:
        :param coordinates:
        :param central_carbon_nbd_idx:
        :param hydrogens_bonded_to_carbon_back_nbd_idxs:
        :return:
        """
        elements = convert_elements(elements, 'symbols')
        metal_idx = metal_idx - 1
        bidentate_min_donor_idx = bidentate_min_donor_idx - 1
        bidentate_max_donor_idx = bidentate_max_donor_idx - 1
        metal_coordinates = coordinates[metal_idx]
        bidentate_min_donor_coordinates = coordinates[bidentate_min_donor_idx]
        bidentate_max_donor_coordinates = coordinates[bidentate_max_donor_idx]

        # get the central carbon and the two hydrogens bonded to it
        central_carbon_coordinates = coordinates[central_carbon_nbd_idx]
        hydrogen_1_idx = hydrogens_bonded_to_carbon_back_nbd_idxs[0]
        hydrogen_2_idx = hydrogens_bonded_to_carbon_back_nbd_idxs[1]
        hydrogen_1_coordinates = coordinates[hydrogen_1_idx]
        hydrogen_2_coordinates = coordinates[hydrogen_2_idx]

        # for each hydrogen first find closest donor, then calculate dihedral angle
        # p0 is one of the donors, p1 is the metal center, p2 is the nbd central carbon, p3 is one of the hydrogens bound to p2
        # hydrogen 1
        distance_to_min_donor = calculate_distance(hydrogen_1_coordinates, bidentate_min_donor_coordinates)
        distance_to_max_donor = calculate_distance(hydrogen_1_coordinates, bidentate_max_donor_coordinates)
        if distance_to_min_donor < distance_to_max_donor:
            closest_donor_coordinates = bidentate_min_donor_coordinates
            closest_donor_index = bidentate_min_donor_idx
            dihedral_angle_1 = calculate_dihedral(closest_donor_coordinates, metal_coordinates, central_carbon_coordinates, hydrogen_1_coordinates)
        else:
            closest_donor_coordinates = bidentate_max_donor_coordinates
            closest_donor_index = bidentate_max_donor_idx
            dihedral_angle_1 = calculate_dihedral(closest_donor_coordinates, metal_coordinates, central_carbon_coordinates, hydrogen_1_coordinates)

        # add to dictionary
        dictionary["dihedral_angle_1"] = dihedral_angle_1
        dictionary["dihedral_angle_1_element_1"] = elements[closest_donor_index]
        dictionary["dihedral_angle_1_element_2"] = elements[metal_idx]
        dictionary["dihedral_angle_1_element_3"] = elements[central_carbon_nbd_idx]
        dictionary["dihedral_angle_1_element_4"] = elements[hydrogen_1_idx]
        dictionary["dihedral_angle_1_index_1"] = closest_donor_index + 1
        dictionary["dihedral_angle_1_index_2"] = metal_idx + 1
        dictionary["dihedral_angle_1_index_3"] = central_carbon_nbd_idx + 1
        dictionary["dihedral_angle_1_index_4"] = hydrogen_1_idx + 1

        # hydrogen 2
        distance_to_min_donor = calculate_distance(hydrogen_2_coordinates, bidentate_min_donor_coordinates)
        distance_to_max_donor = calculate_distance(hydrogen_2_coordinates, bidentate_max_donor_coordinates)
        if distance_to_min_donor < distance_to_max_donor:
            closest_donor_coordinates = bidentate_min_donor_coordinates
            closest_donor_index = bidentate_min_donor_idx
            dihedral_angle_2 = calculate_dihedral(closest_donor_coordinates, metal_coordinates, central_carbon_coordinates, hydrogen_2_coordinates)
        else:
            closest_donor_coordinates = bidentate_max_donor_coordinates
            closest_donor_index = bidentate_max_donor_idx
            dihedral_angle_2 = calculate_dihedral(closest_donor_coordinates, metal_coordinates, central_carbon_coordinates, hydrogen_2_coordinates)

        dictionary["dihedral_angle_2"] = dihedral_angle_2
        dictionary["dihedral_angle_2_element_1"] = elements[closest_donor_index]
        dictionary["dihedral_angle_2_element_2"] = elements[metal_idx]
        dictionary["dihedral_angle_2_element_3"] = elements[central_carbon_nbd_idx]
        dictionary["dihedral_angle_2_element_4"] = elements[hydrogen_2_idx]
        dictionary["dihedral_angle_2_index_1"] = closest_donor_index + 1
        dictionary["dihedral_angle_2_index_2"] = metal_idx + 1
        dictionary["dihedral_angle_2_index_3"] = central_carbon_nbd_idx + 1
        dictionary["dihedral_angle_2_index_4"] = hydrogen_2_idx + 1

        return dictionary

    def _buried_volume_quadrant_analysis(self, filename, elements, coordinates, dictionary, metal_idx, z_axis_atoms, xz_plane_atoms, excluded_atoms=None, plot_steric_map=False):
        """
        Calculate the buried volume for the 4 quadrants and 8 octants (positive Z direction) of the bidentate ligand.
        :param elements:
        :param coordinates:
        :param dictionary:
        :param metal_idx:
        :param z_axis_atoms:
        :param xz_plane_atoms:
        :param excluded_atoms:
        :return:
        """
        buried_volume = BuriedVolume(elements, coordinates, metal_idx,
                                                  z_axis_atoms=z_axis_atoms,
                                                  xz_plane_atoms=xz_plane_atoms,
                                                  excluded_atoms=excluded_atoms,
                                                  radius=7).octant_analysis()
        buried_volume_for_quad_oct = buried_volume.octant_analysis()

        quadrants = buried_volume_for_quad_oct.quadrants['percent_buried_volume']
        octants = buried_volume_for_quad_oct.octants['percent_buried_volume']
        quadrant_dictionary = {1: 'NE', 2: 'NW', 3: 'SW', 4: 'SE'}
        octant_dictionary = {0: '+,+,+', 1: '-,+,+', 2: '-,-,+', 3: '+,-,+', 4: '+,-,-', 5: '-,-,-', 6: '-,+,-',
                             7: '+,+,-'}

        for quad_index in range(4):
            values = list(quadrants.values())
            dictionary[quadrant_dictionary[quad_index + 1] + "_quad"] = values[quad_index] / 100

        for oct_index in range(8):
            values = list(octants.values())
            dictionary[octant_dictionary[oct_index] + "_octant"] = values[oct_index] / 100
        if plot_steric_map:
            buried_volume.plot_steric_map(filename=os.path.join(self.path_to_workflow, filename + '_steric_map.png'))
        return dictionary

    def _merge_descriptor_dfs(self, old_descriptor_df, new_descriptor_df):
        """
        When self.descriptor_df is not None, the new descriptor df needs to be merged with the old one

        :param old_descriptor_df:
        :param new_descriptor_df:
        :return:
        """
        old_descriptor_df = old_descriptor_df.merge(new_descriptor_df, on=['filename_tud',
                                                                            f"index_{self.central_atom}",
                                                                            "index_donor_max",
                                                                            "index_donor_min",
                                                                            f"element_{self.central_atom}",
                                                                            "element_donor_max",
                                                                            "element_donor_min"], how='left')
        return old_descriptor_df

    def set_output_type(self, new_output_type):
        """
        Set the output type of the descriptor calculator. In this way you can calculate descriptors on CREST output
        first and the xtb xyz's afterwards (or other way around).

        :param new_output_type:
        :return:
        """
        if new_output_type not in self.supported_output_types:
            raise ValueError(f'Output type {new_output_type} not supported. Please choose from {self.supported_output_types}.')
        self.output_type = new_output_type

    def _calculate_steric_electronic_desc_morfeus(self, solvent, dictionary, elements, coordinates, filename, metal_adduct='pristine', plot_steric_map=False, bidentate_min_donor_idx=None, bidentate_max_donor_idx=None):
        """
        Calculate all steric and electronic descriptors that can be calculated using Morfeus. For NBD ligands,
        there are additional descriptors that can be calculated.

        :param bidentate_min_donor_idx: if user does not want to use xTB for this, they can give the max/min donor idx
        :param bidentate_max_donor_idx: if user does not want to use xTB for this, they can give the max/min donor idx
        :param solvent:
        :param dictionary:
        :param elements:
        :param coordinates:
        :param filename:
        :param metal_adduct:
        :return:
        """
        # first we need to find the bidentate ligand atoms and the metal center
        # bidentate_ligand_atoms = all indices of the bidentate ligand + metal center
        # bidentate = just indices of the metal center and the two bidentate ligand atoms
        bidentate_ligand_atoms, bidentate = self._find_bidentate_ligand(elements, coordinates)
        # get elements and coordinates for the bidentate ligand atoms only, these are used in cone angle and quadrant
        # calculations
        bidentate_ligand_atoms_coordinates = coordinates[bidentate_ligand_atoms]
        bidentate_ligand_atoms_elements = elements[bidentate_ligand_atoms]
        # metal center idx is always at end of molecular_graph output, so we search for its coordinates to find the index
        bidentate_ligand_atoms_metal_idx = np.where(np.array(bidentate_ligand_atoms_coordinates) == coordinates[bidentate_ligand_atoms[-1]])[0][0] + 1
        # print(bidentate_ligand_atoms_elements[bidentate_ligand_atoms_metal_idx - 1])
        # first index is the metal, second index is the bidentate ligand 1, third index is the bidentate ligand 2
        # in the full complex
        # morfeus indices start at 1, so add 1 to the indices
        metal_idx = bidentate[0] + 1
        bidentate_1_idx = bidentate[1] + 1
        bidentate_2_idx = bidentate[2] + 1

        # determine max or min donor based on xTB charge of the donor atoms
        # ToDo: make this optional, so that the user can choose whether to use xTB for this or give the max/min donor
        xtb_functional = 2  # indicate whether GFN1 or GFN2 is used for electronic descriptors
        if xtb_functional == 1:
            calculation_method = 'gfn1_xtb'
        else:
            calculation_method = 'gfn2_xtb'

        xtb = morfeus.XTB(elements, coordinates, solvent=solvent)
        atomic_charges = xtb.get_charges()  # determine min/max donor based on xTB charge of the donor atoms
        charge_bidentate_1 = atomic_charges[bidentate_1_idx]
        charge_bidentate_2 = atomic_charges[bidentate_2_idx]

        if bidentate_max_donor_idx is None or bidentate_min_donor_idx is None:
            # if the user has not given the max/min donor indices, we determine them based on the xTB charges
            # the user can give the indices for example based on a structure template where the indices stay the same
            # or on DFT charges etc.
            if charge_bidentate_1 > charge_bidentate_2:
                # this means that bidentate 1 is max donor (charge is less negative, so stronger donor)
                bidentate_max_donor_idx = bidentate_1_idx
                bidentate_min_donor_idx = bidentate_2_idx
            else:
                # this means that bidentate 2 is max donor
                bidentate_max_donor_idx = bidentate_2_idx
                bidentate_min_donor_idx = bidentate_1_idx

        bidentate_ligand_atoms_max_donor_idx = np.where(np.array(bidentate_ligand_atoms_coordinates) == coordinates[bidentate_max_donor_idx - 1])[0][0] + 1
        bidentate_ligand_atoms_min_donor_idx = np.where(np.array(bidentate_ligand_atoms_coordinates) == coordinates[bidentate_min_donor_idx - 1])[0][0] + 1
        # print(bidentate_ligand_atoms_elements[bidentate_ligand_atoms_max_donor_idx - 1])
        # print(bidentate_ligand_atoms_elements[bidentate_ligand_atoms_min_donor_idx - 1])

        # write indices and elements of metal center, max donor, and min donor to dictionary
        dictionary[f"index_{self.central_atom}"] = metal_idx
        dictionary["index_donor_max"] = bidentate_max_donor_idx
        dictionary["index_donor_min"] = bidentate_min_donor_idx
        element_symbols = convert_elements(elements, output='symbols')
        dictionary[f"element_{self.central_atom}"] = element_symbols[metal_idx - 1]
        dictionary["element_donor_max"] = element_symbols[bidentate_max_donor_idx - 1]
        dictionary["element_donor_min"] = element_symbols[bidentate_min_donor_idx - 1]

        # by default no atoms are excluded from buried volume analysis and all elements are taken for cone angle analysis
        # parameters for quadrant analysis
        excluded_atoms = None
        # z_axis_atom_index = [bidentate_min_donor_idx,
        #                      bidentate_max_donor_idx]  # the index in the log file is 0-based, but the index in morfeus is 1-based
        # xz_plane_atom_indices = [bidentate_max_donor_idx]

        # parameters for cone angle
        # cone_angle_elements = elements
        # cone_angle_coordinates = coordinates
        # cone_angle_correct = True  # whether we can proceed with the cone angle calculation later or not

        # if the metal adduct is NBD we can do buried volume quadrant analysis, calculate dihedral angles and C=C bond lengths
        if metal_adduct.lower() == 'nbd':
            # C=C bond distance (indicator of amount of pi donation to metal center)
            dictionary.update(self._calculate_c_c_distance_nbd(elements, coordinates, dictionary))

            nbd_complex = NBDComplex(elements, coordinates, filename)
            # dihedral angles
            carbon_back_nbd_idx = nbd_complex.check_nbd_back_carbon()
            hydrogens_bonded_to_carbon_back_nbd = nbd_complex.get_hydrogens_bonded_to_carbon_back_nbd()
            # if this is not the case, it means that the NBD is not in the correct orientation in the complex
            if carbon_back_nbd_idx is not None and hydrogens_bonded_to_carbon_back_nbd is not None:
                print('NBD found at end of xyz file, calculating stuff the easy way')
                dictionary.update(
                    self._calculate_dihedral_angles_nbd_and_metal_donors(dictionary, metal_idx,
                                                                         bidentate_max_donor_idx,
                                                                         bidentate_min_donor_idx, elements,
                                                                         coordinates, carbon_back_nbd_idx,
                                                                         hydrogens_bonded_to_carbon_back_nbd))

                # quadrant analysis parameters
                # exclude all nbd atoms from the quadrant analysis
                # last 15 atoms are the nbd atoms, but indexing in morfeus is 1-based
                # nbd_indices = list(range(len(elements) - 15, len(elements) + 1))
                # excluded_atoms = nbd_indices

                # if everything with NBD is fine, we can delete all auxillary ligands and calculate the cone angle
                # cone_angle_correct = True
                # cone_angle_elements = elements[:-15]
                # cone_angle_coordinates = coordinates[:-15]

            elif carbon_back_nbd_idx is None and hydrogens_bonded_to_carbon_back_nbd is None:
                print('NBD not found at end of xyz file, calculating stuff the hard way')
                carbon_back_nbd_and_hydrogens_idx = nbd_complex.find_central_carbon_and_hydrogens_nbd_openbabel()
                if carbon_back_nbd_and_hydrogens_idx is not None:
                    carbon_back_nbd_idx = carbon_back_nbd_and_hydrogens_idx[0]
                    hydrogens_bonded_to_carbon_back_nbd = [carbon_back_nbd_and_hydrogens_idx[1], carbon_back_nbd_and_hydrogens_idx[2]]
                    # try molsimplify + openbabel approach for identifying the NBD
                    dictionary.update(
                        self._calculate_dihedral_angles_nbd_and_metal_donors(dictionary, metal_idx,
                                                                             bidentate_max_donor_idx,
                                                                             bidentate_min_donor_idx, elements,
                                                                             coordinates, carbon_back_nbd_idx,
                                                                             hydrogens_bonded_to_carbon_back_nbd))

                    # quadrant analysis parameters
                    # ToDo: fix nbd_complex.find_nbd_openbabel() such that we can remove NBD for quadrant analysis
                # cone_angle_correct = False  # ToDo: fix nbd_complex.find_nbd_openbabel() such that we can remove NBD for cone angle calculation

        # for the quadrant analysis, we only use the bidentate ligand atoms and metal center

        z_axis_atom_index = [bidentate_ligand_atoms_min_donor_idx, bidentate_ligand_atoms_max_donor_idx]
        xz_plane_atom_indices = [bidentate_ligand_atoms_max_donor_idx]
        dictionary.update(
            self._buried_volume_quadrant_analysis(filename, bidentate_ligand_atoms_elements, bidentate_ligand_atoms_coordinates, dictionary, bidentate_ligand_atoms_metal_idx,
                                                  z_axis_atom_index, xz_plane_atom_indices, excluded_atoms, plot_steric_map))


        # calculate steric descriptors
        # for the bite angle we can use the whole complex
        try:
            bite_angle = BiteAngle(coordinates, metal_idx, bidentate_1_idx,
                                                 bidentate_2_idx).angle  # unit: degrees
            dictionary["bite_angle"] = bite_angle
            # print('Bite angle: {}'.format(bite_angle)
        except Exception:
            print('Bite angle calculation failed, defaulting to None.')
            dictionary["bite_angle"] = None

        # for the cone angle we again only use the bidentate ligand atoms and metal center
        try:
            cone_angle = ConeAngle(bidentate_ligand_atoms_elements, bidentate_ligand_atoms_coordinates,
                                   bidentate_ligand_atoms_metal_idx).cone_angle
            dictionary["cone_angle"] = cone_angle
            # print('Cone angle: {}'.format(cone_angle)
        except Exception:
            print('Cone angle calculation failed, defaulting to None.')
            dictionary["cone_angle"] = None

        # if cone_angle_correct:
        #     if geom_type == "BD" or geom_type == "SP":
        #         try:
        #             cone_angle = ConeAngle(bidentate_ligand_atoms_elements, bidentate_ligand_atoms_coordinates,
        #                                                  bidentate_ligand_atoms_metal_idx).cone_angle
        #             dictionary["cone_angle"] = cone_angle
        #             # print('Cone angle: {}'.format(cone_angle)
        #         except Exception:
        #             print('Cone angle calculation failed, defaulting to None.')
        #             dictionary["cone_angle"] = None
        #     else:
        #         try:
        #             a = [bidentate[0]]
        #             a.extend(ligand_atoms[bidentate[1]])
        #             a = list(np.sort(np.array(a)))
        #
        #         except Exception:
        #             print('Molecular graph search failed, defaulting to manual search.')
        #             a = list(np.sort(np.array(bidentate)))
        #
        #         diff = None
        #         for id, i in enumerate(a):
        #             if i == bidentate[0]:
        #                 diff = id
        #         elements_cone_angle = cone_angle_elements[a]
        #         coordinates_cone_angle = np.array(cone_angle_coordinates)[a]
        #         if diff is not None:
        #             try:
        #                 dictionary["cone_angle"] = ConeAngle(elements_cone_angle, coordinates_cone_angle,
        #                                                      diff + 1).cone_angle
        #             except Exception:
        #                 # unit: degrees
        #                 print('Cone angle calculation failed, defaulting to None. For complex:', complex)
        #                 dictionary["cone_angle"] = None
        #
        # else:
        #     dictionary["cone_angle"] = None

        # calculate distances between metal and donors, units: angstrom

        if self.output_type is None:
            dictionary[f"distance_{self.central_atom}_max_donor_"] = calculate_distance(coordinates[metal_idx - 1], coordinates[bidentate_max_donor_idx - 1])
            dictionary[f"distance_{self.central_atom}_min_donor_"] = calculate_distance(coordinates[metal_idx - 1], coordinates[bidentate_min_donor_idx - 1])
        else:
            dictionary[f"distance_{self.central_atom}_max_donor_{self.output_type.lower()}"] = calculate_distance(coordinates[metal_idx - 1], coordinates[bidentate_max_donor_idx - 1])
            dictionary[f"distance_{self.central_atom}_min_donor_{self.output_type.lower()}"] = calculate_distance(coordinates[metal_idx - 1], coordinates[bidentate_min_donor_idx - 1])

        # calculate buried volume descriptors on only the bidentate ligand atoms
        bv_metal_center = BuriedVolume(bidentate_ligand_atoms_elements, bidentate_ligand_atoms_coordinates, bidentate_ligand_atoms_metal_idx, radius=3.5, excluded_atoms=excluded_atoms).fraction_buried_volume
        bv_max_donor = BuriedVolume(bidentate_ligand_atoms_elements, bidentate_ligand_atoms_coordinates, bidentate_ligand_atoms_max_donor_idx, radius=3.5, excluded_atoms=excluded_atoms).fraction_buried_volume
        bv_min_donor = BuriedVolume(bidentate_ligand_atoms_elements, bidentate_ligand_atoms_coordinates, bidentate_ligand_atoms_min_donor_idx, radius=3.5, excluded_atoms=excluded_atoms).fraction_buried_volume
        # unit: fraction of volume occupied by atoms within 3.5A of the atom
        dictionary[f"buried_volume_{self.central_atom}_3.5A"] = bv_metal_center
        dictionary["buried_volume_donor_max"] = bv_max_donor
        dictionary["buried_volume_donor_min"] = bv_min_donor

        bv_metal_4 = BuriedVolume(bidentate_ligand_atoms_elements, bidentate_ligand_atoms_coordinates, bidentate_ligand_atoms_metal_idx, radius=4, excluded_atoms=excluded_atoms).fraction_buried_volume
        bv_metal_5 = BuriedVolume(bidentate_ligand_atoms_elements, bidentate_ligand_atoms_coordinates, bidentate_ligand_atoms_metal_idx, radius=5, excluded_atoms=excluded_atoms).fraction_buried_volume
        bv_metal_6 = BuriedVolume(bidentate_ligand_atoms_elements, bidentate_ligand_atoms_coordinates, bidentate_ligand_atoms_metal_idx, radius=6, excluded_atoms=excluded_atoms).fraction_buried_volume
        bv_metal_7 = BuriedVolume(bidentate_ligand_atoms_elements, bidentate_ligand_atoms_coordinates, bidentate_ligand_atoms_metal_idx, radius=7, excluded_atoms=excluded_atoms).fraction_buried_volume

        dictionary[f"buried_volume_{self.central_atom}_4A"] = bv_metal_4
        dictionary[f"buried_volume_{self.central_atom}_5A"] = bv_metal_5
        dictionary[f"buried_volume_{self.central_atom}_6A"] = bv_metal_6
        dictionary[f"buried_volume_{self.central_atom}_7A"] = bv_metal_7

        # print(BuriedVolume(ce.elements, conformer.coordinates, bidentate[1]).print_report())
        # BuriedVolume(elements, coordinates, bidentate[1], radius=4).print_report()

        # calculate dispersion, SASA and electronic descriptors using morfeus
        # these are calculated on the whole complex
        # dispersion P_int unit: kcal^0.5 mol^-0.5
        dictionary[f"dispersion_p_int_{self.central_atom}_{calculation_method}"] = \
            Dispersion(elements, coordinates).atom_p_int[metal_idx]
        dictionary[f"dispersion_p_int_donor_max_{calculation_method}"] = \
            Dispersion(elements, coordinates).atom_p_int[bidentate_max_donor_idx]
        dictionary[f"dispersion_p_int_donor_min_{calculation_method}"] = \
            Dispersion(elements, coordinates).atom_p_int[bidentate_min_donor_idx]
        # SASA unit: A^2
        dictionary[f"sasa_{calculation_method}"] = SASA(elements, coordinates).area

        dictionary[f"ip_{calculation_method}"] = xtb.get_ip(corrected=True)  # unit: eV
        dipole = xtb.get_dipole()
        dictionary[f"dipole_{calculation_method}"] = np.sqrt(dipole.dot(dipole))  # unit: Debye dipole moment
        dictionary[f"ea_{calculation_method}"] = xtb.get_ea()  # unit: eV
        dictionary[f"electrofugality_{calculation_method}"] = xtb.get_global_descriptor(
            variety='electrofugality', corrected=True)  # unit: eV
        dictionary[f"nucleofugality_{calculation_method}"] = xtb.get_global_descriptor(
            variety='nucleofugality', corrected=True)  # unit: eV
        dictionary[f"nucleophilicity_{calculation_method}"] = xtb.get_global_descriptor(
            variety='nucleophilicity', corrected=True)  # unit: eV
        dictionary[f"electrophilicity_{calculation_method}"] = xtb.get_global_descriptor(
            variety='electrophilicity', corrected=True)  # unit: eV

        homo = xtb.get_homo()
        lumo = xtb.get_lumo()
        dictionary[f"HOMO_LUMO_gap_{calculation_method}"] = lumo - homo
        
        return dictionary, metal_idx, bidentate_max_donor_idx, bidentate_min_donor_idx

    def _calculate_dft_descriptors_from_log(self, dft, dictionary):
        """
        Calculate descriptors from DFT log file using the DFTExtractor class.

        :param log_file:
        :param metal_idx:
        :param bidentate_max_donor_idx:
        :param bidentate_min_donor_idx:
        :param dictionary:
        :param metal_adduct:
        :return:
        """
        successful_dft_optimization = dft.check_normal_termination()
        dictionary["optimization_success_dft"] = successful_dft_optimization
        wall_time, cpu_time = dft.extract_time()
        # the last item of wall_time and cpu_time in cclib are the actual extracted values
        wall_time = wall_time[-1]
        cpu_time = cpu_time[-1]
        # convert timedelta object to hour duration for easy comparison
        wall_time = float(wall_time.total_seconds()) / 3600
        cpu_time = float(cpu_time.total_seconds()) / 3600
        dictionary["wall_time_dft"] = wall_time
        dictionary["cpu_time_dft"] = cpu_time

        if successful_dft_optimization:
            # if metal_adduct.lower() == "nbd":
                # these are calculated in _calculate_steric_electronic_desc_morfeus already
                # # quadrant analysis
                # z_axis_atom_index = dft.check_nbd_back_carbon()
                # if z_axis_atom_index is not None:  # if the NBD carbon is found it is safe to proceed
                #     z_axis_atom_index += 1  # the index in the log file is 0-based, but the index in morfeus is 1-based
                #     xz_plane_atom_indices = [bidentate_min_donor_idx, bidentate_max_donor_idx, metal_idx]
                #     # exclude all nbd atoms from the quadrant analysis
                #     # last 15 atoms are the nbd atoms, but indexing in morfeus is 1-based
                #     nbd_indices = list(range(len(dft.elements) - 15, len(dft.elements) + 1))
                #     dictionary.update(self._buried_volume_quadrant_analysis(dft.elements, dft.coordinates, dictionary, metal_idx, z_axis_atom_index, xz_plane_atom_indices, nbd_indices))

                    # # C=C bond distance (indicator of amount of pi donation to metal center)
                    # dictionary.update(self._calculate_c_c_distance_nbd(dft.elements, dft.coordinates, dictionary))
                    #
                    # # dihedral angles
                    # carbon_back_nbd_idx = dft.check_nbd_back_carbon()
                    # hydrogens_bonded_to_carbon_back_nbd = dft.get_hydrogens_bonded_to_carbon_back_nbd()
                    # if carbon_back_nbd_idx is not None and hydrogens_bonded_to_carbon_back_nbd is not None:
                    #     dictionary.update(self._calculate_dihedral_angles_nbd_and_metal_donors(dictionary, metal_idx, bidentate_max_donor_idx, bidentate_min_donor_idx, dft.elements, dft.coordinates, carbon_back_nbd_idx, hydrogens_bonded_to_carbon_back_nbd))

            # thermodynamic descriptors
            sum_electronic_and_free_energy, sum_electronic_and_enthalpy, zero_point_correction, entropy = dft.extract_thermodynamic_descriptors()
            dictionary[f"sum_electronic_and_free_energy_dft"] = sum_electronic_and_free_energy
            dictionary[f"sum_electronic_and_enthalpy_dft"] = sum_electronic_and_enthalpy
            dictionary[f"zero_point_correction_dft"] = zero_point_correction
            dictionary[f"entropy_dft"] = entropy

            # orbital occupations
            # donor with metal
            if self.central_atom is not None:
                # this will be skipped for the free ligand since there is no metal center there
                min_donor_metal_orbital_occupation, min_donor_metal_anti_orbital_occupation = dft.calculate_min_donor_metal_orbital_occupation(), dft.calculate_min_donor_metal_anti_orbital_occupation()
                dictionary[f"orbital_occupation_min_donor_{self.central_atom}_dft"] = min_donor_metal_orbital_occupation
                dictionary[f"anti_orbital_occupation_min_donor_{self.central_atom}_dft"] = min_donor_metal_anti_orbital_occupation
                max_donor_metal_orbital_occupation, max_donor_metal_anti_orbital_occupation = dft.calculate_max_donor_metal_orbital_occupation(), dft.calculate_max_donor_metal_anti_orbital_occupation()
                dictionary[f"orbital_occupation_max_donor_{self.central_atom}_dft"] = max_donor_metal_orbital_occupation
                dictionary[f"anti_orbital_occupation_max_donor_{self.central_atom}_dft"] = max_donor_metal_anti_orbital_occupation

            # donor with any other element
            min_donor_other_element_index_list, min_donor_other_orbital_occupation_list = dft.calculate_min_donor_other_orbital_occupation()
            if not min_donor_other_element_index_list is None and not min_donor_other_orbital_occupation_list is None:
                for i, (element_and_index, occupation) in enumerate(zip(min_donor_other_element_index_list, min_donor_other_orbital_occupation_list)):
                    other_element = element_and_index[0]
                    other_element_index = element_and_index[1]
                    dictionary[f"orbital_occupation_min_donor_other_atom_{i + 1}_dft"] = occupation
                    dictionary[f"orbital_occupation_min_donor_other_atom_{i + 1}_element_dft"] = other_element
                    dictionary[f"orbital_occupation_min_donor_other_atom_{i + 1}_index_dft"] = other_element_index
            else:
                dictionary[f"orbital_occupation_min_donor_other_atom_1_dft"] = None
                dictionary[f"orbital_occupation_min_donor_other_atom_1_element_dft"] = None
                dictionary[f"orbital_occupation_min_donor_other_atom_1_index_dft"] = None

            min_donor_other_element_index_anti_bonding_list, min_donor_other_anti_orbital_occupation_list = dft.calculate_min_donor_other_anti_orbital_occupation()
            if not min_donor_other_element_index_anti_bonding_list is None and not min_donor_other_anti_orbital_occupation_list is None:
                for i, (element_and_index, occupation) in enumerate(zip(min_donor_other_element_index_anti_bonding_list, min_donor_other_anti_orbital_occupation_list)):
                    other_element = element_and_index[0]
                    other_element_index = element_and_index[1]
                    dictionary[f"anti_orbital_occupation_min_donor_other_atom_{i + 1}_dft"] = occupation
                    dictionary[f"anti_orbital_occupation_min_donor_other_atom_{i + 1}_element_dft"] = other_element
                    dictionary[f"anti_orbital_occupation_min_donor_other_atom_{i + 1}_index_dft"] = other_element_index
            else:
                dictionary[f"anti_orbital_occupation_min_donor_other_atom_1_dft"] = None
                dictionary[f"anti_orbital_occupation_min_donor_other_atom_1_element_dft"] = None
                dictionary[f"anti_orbital_occupation_min_donor_other_atom_1_index_dft"] = None

            max_donor_other_element_index_list, max_donor_other_orbital_occupation_list = dft.calculate_max_donor_other_orbital_occupation()
            if not max_donor_other_element_index_list is None and not max_donor_other_orbital_occupation_list is None:
                for i, (element_and_index, occupation) in enumerate(zip(max_donor_other_element_index_list, max_donor_other_orbital_occupation_list)):
                    other_element = element_and_index[0]
                    other_element_index = element_and_index[1]
                    dictionary[f"orbital_occupation_max_donor_other_atom_{i + 1}_dft"] = occupation
                    dictionary[f"orbital_occupation_max_donor_other_atom_{i + 1}_element_dft"] = other_element
                    dictionary[f"orbital_occupation_max_donor_other_atom_{i + 1}_index_dft"] = other_element_index
            else:
                dictionary[f"orbital_occupation_max_donor_other_atom_1_dft"] = None
                dictionary[f"orbital_occupation_max_donor_other_atom_1_element_dft"] = None
                dictionary[f"orbital_occupation_max_donor_other_atom_1_index_dft"] = None

            max_donor_other_element_index_anti_bonding_list, max_donor_other_anti_orbital_occupation_list = dft.calculate_max_donor_other_anti_orbital_occupation()
            if not max_donor_other_element_index_anti_bonding_list is None and not max_donor_other_anti_orbital_occupation_list is None:
                for i, (element_and_index, occupation) in enumerate(zip(max_donor_other_element_index_anti_bonding_list, max_donor_other_anti_orbital_occupation_list)):
                    other_element = element_and_index[0]
                    other_element_index = element_and_index[1]
                    dictionary[f"anti_orbital_occupation_max_donor_other_atom_{i + 1}_dft"] = occupation
                    dictionary[f"anti_orbital_occupation_max_donor_other_atom_{i + 1}_element_dft"] = other_element
                    dictionary[f"anti_orbital_occupation_max_donor_other_atom_{i + 1}_index_dft"] = other_element_index
            else:
                dictionary[f"anti_orbital_occupation_max_donor_other_atom_1_dft"] = None
                dictionary[f"anti_orbital_occupation_max_donor_other_atom_1_element_dft"] = None
                dictionary[f"anti_orbital_occupation_max_donor_other_atom_1_index_dft"] = None

            # dipole moment
            dipole_moment = dft.calculate_dipole_moment()
            dictionary[f"dipole_moment_dft"] = dipole_moment

            # lone pair occupancy
            lone_pair_occupancy_min_donor, lone_pair_occupancy_max_donor = dft.calculate_donor_lone_pair_occupancy()
            dictionary["lone_pair_occupancy_min_donor_dft"] = lone_pair_occupancy_min_donor
            dictionary["lone_pair_occupancy_max_donor_dft"] = lone_pair_occupancy_max_donor

            # dispersion energy
            dispersion_energy = dft.calculate_dispersion_energy()
            dictionary["dispersion_energy_dft"] = dispersion_energy

            # NBO charges
            metal_nbo_charge, min_donor_nbo_charge, max_donor_nbo_charge = dft.calculate_natural_charges()
            if self.central_atom is not None:
                # this will be skipped for the free ligand since there is no metal center there
                dictionary[f"nbo_charge_{self.central_atom}_dft"] = metal_nbo_charge
            dictionary[f"nbo_charge_min_donor_dft"] = min_donor_nbo_charge
            dictionary[f"nbo_charge_max_donor_dft"] = max_donor_nbo_charge

            # mulliken charges
            metal_mulliken_charge, min_donor_mulliken_charge, max_donor_mulliken_charge = dft.calculate_mulliken_charges()
            if self.central_atom is not None:
                # this will be skipped for the free ligand since there is no metal center there
                dictionary[f"mulliken_charge_{self.central_atom}_dft"] = metal_mulliken_charge
            dictionary[f"mulliken_charge_min_donor_dft"] = min_donor_mulliken_charge
            dictionary[f"mulliken_charge_max_donor_dft"] = max_donor_mulliken_charge

            # other electronic descriptors
            homo_energy, lumo_energy, homo_lumo_gap, hardness, softness, electronegativity, electrophilicity = dft.calculate_electronic_descriptors()
            dictionary["homo_energy_dft"] = homo_energy
            dictionary["lumo_energy_dft"] = lumo_energy
            dictionary["homo_lumo_gap_dft"] = homo_lumo_gap
            dictionary["hardness_dft"] = hardness
            dictionary["softness_dft"] = softness
            dictionary["electronegativity_dft"] = electronegativity
            dictionary["electrophilicity_dft"] = electrophilicity

        return dictionary

    def calculate_morfeus_descriptors(self, solvent=None, printout=False, metal_adduct='pristine', plot_steric_map=False):
        """
        Function that creates the dictionary for descriptor calculation using Morfeus and performs the right actions
        based on the output type. For CREST ensembles, the descriptors are boltzmann weighted and averaged.

        :param solvent:
        :param printout:
        :param metal_adduct:
        :return:
        """
        if self.output_type.lower() == 'xyz':
            complexes_to_calc_descriptors = glob.glob(os.path.join(self.path_to_workflow, '*.xyz'))
            dictionary_for_properties = {}

            # try:
            for metal_ligand_complex in tqdm(complexes_to_calc_descriptors):
                properties = {}

                base_with_extension = os.path.basename(metal_ligand_complex)
                split_base = os.path.splitext(base_with_extension)
                filename = split_base[0]
                print('\nCalculating descriptors for: ', filename, '...')
                properties['filename_tud'] = filename

                elements, coordinates = read_xyz(metal_ligand_complex)
                try:
                    properties, metal_idx, bidentate_max_donor_idx, bidentate_min_donor_idx = self._calculate_steric_electronic_desc_morfeus(solvent=solvent, dictionary=properties, elements=elements, coordinates=coordinates, filename=filename, metal_adduct=metal_adduct, plot_steric_map=plot_steric_map)
                except:
                    # if something goes wrong with the molecular graph it usually means that the geometry is wrong
                    print('Error calculating Morfeus descriptors for: ', filename)
                    print('Make sure to check the geometry')
                dictionary_for_properties[os.path.basename(os.path.normpath(metal_ligand_complex[:-4]))] = properties

            new_descriptor_df = dataframe_from_dictionary(dictionary_for_properties)

            if printout:
                print(new_descriptor_df.to_markdown())

            # merge descriptor dataframes if they already exist
            if self.descriptor_df is None:
                self.descriptor_df = new_descriptor_df
            else:
                self.descriptor_df = self._merge_descriptor_dfs(self.descriptor_df, new_descriptor_df)

        elif self.output_type.lower() == 'crest':
            complexes_to_calc_descriptors = glob.glob(os.path.join(self.path_to_workflow, 'CREST', '*'))
            dictionary_for_conformer_properties = {}
            for complex in tqdm(complexes_to_calc_descriptors):
                conformer_properties = {}
                #ce = None
                filename = os.path.basename(os.path.normpath(complex))
                print('\nCalculating descriptors for: ', filename, '...')
                try:
                    self.ce = ConformerEnsemble.from_crest(complex)
                except Exception as e:
                    print("Descriptor calculation failed for this complex:", complex)
                    print(e)
                    conformer_properties['filename_tud'] = filename
                    continue
                self.calculate_morfeus_descriptors_ce(metal_adduct = metal_adduct, ce_name=filename)
            #     if ce is not None:
            #         # ce.generate_mol()
            #         ce.prune_energy()
            #         ce.sort()
            #         conformer_properties['filename_tud'] = filename
            #         print(f'Number of conformers in ensemble for {filename}: {len(ce)}')
            #         # define indexing columns that need to be excluded from boltzmann averaging
            #         columns_to_exclude = [f"index_{self.central_atom}", "index_donor_max", "index_donor_min",
            #                               f"element_{self.central_atom}", "element_donor_max",
            #                               "element_donor_min"]
            #         if metal_adduct.lower() == 'nbd':  # in this case there are some additional columns that need to be skipped
            #             columns_to_exclude += ['distance_pi_bond_1_element_1', 'distance_pi_bond_1_element_2',
            #                                    'distance_pi_bond_1_element_1_idx', 'distance_pi_bond_1_element_2_idx',
            #                                    'distance_pi_bond_2_element_1', 'distance_pi_bond_2_element_2',
            #                                    'distance_pi_bond_2_element_1_idx', 'distance_pi_bond_2_element_2_idx',
            #                                    'dihedral_angle_1_element_1', 'dihedral_angle_1_element_2',
            #                                    'dihedral_angle_1_element_3', 'dihedral_angle_1_element_4',
            #                                    'dihedral_angle_1_index_1', 'dihedral_angle_1_index_2',
            #                                    'dihedral_angle_1_index_3', 'dihedral_angle_1_index_4',
            #                                    'dihedral_angle_2_element_1', 'dihedral_angle_2_element_2',
            #                                    'dihedral_angle_2_element_3', 'dihedral_angle_2_element_4',
            #                                    'dihedral_angle_2_index_1', 'dihedral_angle_2_index_2',
            #                                    'dihedral_angle_2_index_3', 'dihedral_angle_2_index_4']

            #         # initialize list which will contain conformers to be removed if indexing properties are not
            #         # the same as the first conformer
            #         # remove_conformer_list = []

            #         try:
            #             for conformer_idx, conformer in enumerate(ce.conformers):
            #                 elements, coordinates = ce.elements, conformer.coordinates
            #                 conformer.properties, metal_idx, bidentate_max_donor_idx, bidentate_min_donor_idx = self._calculate_steric_electronic_desc_morfeus(solvent=solvent, dictionary=conformer.properties, elements=elements, coordinates=coordinates, filename=filename, metal_adduct=metal_adduct, plot_steric_map=plot_steric_map)
            #                 # these are indexing properties so we don't want them to be boltzmann averaged

            #                 # check if indexing column is the same as the first conformer, if not add conformer to
            #                 # remove_conformer_list to be removed from the conformer ensemble later
            #                 bidentate_min_donor_idx_same_across_all_conformers = True
            #                 bidentate_max_donor_idx_same_across_all_conformers = True
            #                 for key in [k for k in ce.get_properties().keys() if k in columns_to_exclude]:
            #                     if conformer.properties[key] != ce.conformers[0].properties[key]:
            #                         print("\n"
            #                             f"BE AWARE: Indexing property {key} is not the same across all conformers for conformer {conformer_idx}.")
            #                             # f"This conformer will be deleted for {os.path.basename(os.path.normpath(complex))}.")
            #                         # ToDo: if this is the case, rerun the conformer's properties with the min/max donor
            #                         # ToDo: indexing of the first conformer
            #                         if key == 'index_donor_min':
            #                             bidentate_min_donor_idx_same_across_all_conformers = False
            #                         elif key == 'index_donor_max':
            #                             bidentate_max_donor_idx_same_across_all_conformers = False
            #                 if not bidentate_min_donor_idx_same_across_all_conformers or not bidentate_max_donor_idx_same_across_all_conformers:
            #                     print(f"BE AWARE: Min and/or max donor indexing properties are not the same across all conformers for {os.path.basename(os.path.normpath(complex))}.")
            #                     print("The min and/or max donor indexing properties will be recalculated for this conformer.")
            #                     print("Using the indexing properties of the first conformer...")
            #                     first_conformer_bidentate_min_donor_idx = ce.conformers[0].properties['index_donor_min']
            #                     first_conformer_bidentate_max_donor_idx = ce.conformers[0].properties['index_donor_max']
            #                     conformer.properties, metal_idx, bidentate_max_donor_idx, bidentate_min_donor_idx = self._calculate_steric_electronic_desc_morfeus(solvent=solvent, dictionary=conformer.properties, elements=elements, coordinates=coordinates, filename=filename, metal_adduct=metal_adduct, plot_steric_map=plot_steric_map, bidentate_max_donor_idx=first_conformer_bidentate_max_donor_idx, bidentate_min_donor_idx=first_conformer_bidentate_min_donor_idx)


            #             #             remove_conformer_list.append(conformer_idx)
            #             #
            #             # # get unique list of conformers to remove and delete them from the conformer ensemble
            #             # remove_conformer_list = list(set(remove_conformer_list))
            #             # for remove_conformer_idx in reversed(remove_conformer_list):
            #             #     del ce.conformers[remove_conformer_idx]
            #             # print(f"\nNumber of conformers in ensemble for {filename} after removing conformers with different indexing properties: {len(ce)}")

            #             # boltzmann averaging
            #             for key in [k for k in ce.get_properties().keys() if k not in columns_to_exclude]:
            #                 conformer_properties[f"{key}_boltzmann_average"] = ce.boltzmann_statistic(key)
            #                 conformer_properties[f"{key}_boltzmann_std"] = ce.boltzmann_statistic(key,
            #                                                                                       statistic='std')
            #                 conformer_properties[f"{key}_boltzmann_variance"] = ce.boltzmann_statistic(key,
            #                                                                                            statistic='var')
            #                 conformer_properties[f"{key}_Emin_conformer"] = ce.get_properties()[key][0]
            #         except Exception as e:
            #             # if something goes wrong with the molecular graph it usually means that the geometry is wrong
            #             print('Error calculating Morfeus descriptors or Boltzmann averaging for: ', filename)
            #             print(e)
            #             print('Make sure to check the geometry')
            #         # all descriptors calculated, now we can write the boltzman statistics to the dictionary
            #     dictionary_for_conformer_properties[os.path.basename(os.path.normpath(complex))] = conformer_properties

            # new_descriptor_df = dataframe_from_dictionary(dictionary_for_conformer_properties)
            # if printout:
            #     print(new_descriptor_df.to_markdown())

            # if self.descriptor_df is None:
            #     self.descriptor_df = new_descriptor_df
            # else:
            #     self.descriptor_df = self._merge_descriptor_dfs(self.descriptor_df, new_descriptor_df)

        else:
            raise ValueError(f'Output type {self.output_type()} not supported. Please choose from {self.supported_output_types}.')

    def calculate_dft_descriptors_from_log(self, solvent=None, extract_xyz_from_log=False, printout=False, metal_adduct='pristine', plot_steric_map=False):
        """
        Function that creates the dictionary and descriptor dataframe for the DFT descriptors. These descriptors are calculated
        from the log files of the DFT calculations. The log files are parsed using the cclib package. The descriptors are
        calculated using either the Morfeus package or extracted from the log files.

        :param solvent:
        :param extract_xyz_from_log:
        :param printout:
        :param metal_adduct:
        :return:
        """
        supported_metal_adducts = ['pristine', 'acetonitrile', 'nbd']  # norbornadiene is placed at bottom of xyz file, so it is a useful pointer for quadrant analysis
        if metal_adduct.lower() not in supported_metal_adducts:
            raise ValueError(f"Metal adduct {metal_adduct} not supported. Please choose from {supported_metal_adducts}.")

        # get all log files
        complexes_to_calc_descriptors = glob.glob(os.path.join(self.path_to_workflow, '*.log'))
        dictionary_for_properties = {}

        # first calculate morfeus descriptors in same way as for xyz files using cclib
        for metal_ligand_complex in tqdm(complexes_to_calc_descriptors):
            properties = {}
            base_with_extension = os.path.basename(metal_ligand_complex)
            split_base = os.path.splitext(base_with_extension)
            filename = split_base[0]
            print('\nCalculating descriptors for:', filename)
            properties['filename_tud'] = filename

            elements, coordinates = None, None
            # error catching for reading elements and coordinates from log file
            try:
                elements, coordinates = read_cclib(metal_ligand_complex)
                if not len(coordinates[-1]) == 3:  # if this is true, there is only 1 coordinates array
                    coordinates = coordinates[-1]  # else morfeus descriptors are calculated for last geometry in log file
                elements = np.array(elements)
            except:
                print('Error reading elements and coordinates from log file for: ', filename)
                print('Make sure to check the geometry')

            # ToDo: initialize DFTExtractor class here and allow assigning min/max donor from the DFT data

            try:
                properties, metal_idx, bidentate_max_donor_idx, bidentate_min_donor_idx = self._calculate_steric_electronic_desc_morfeus(solvent=solvent, dictionary=properties, elements=elements, coordinates=coordinates, filename=filename, metal_adduct=metal_adduct, plot_steric_map=plot_steric_map)
            except Exception as e:
                print('Error calculating Morfeus descriptors for: ', filename)
                print(e)
                print('Make sure to check the geometry')

            # calculate DFT descriptors from Gaussian log file
            # get indices of bidentate ligands and metal for descriptor calculation class
            dft_properties = {}
            try:
                dft = DFTExtractor(metal_ligand_complex, metal_idx, bidentate_min_donor_idx, bidentate_max_donor_idx, metal_adduct)
                dft_properties = self._calculate_dft_descriptors_from_log(dft, dft_properties)
            except Exception as e:
                # print(e)
                print(f'DFT descriptor calculation failed for {filename}')

            if len(dft_properties) > 0:
                properties.update(dft_properties)

            # write xyz for log file
            if extract_xyz_from_log:
                xyz_filename = metal_ligand_complex[:-4] + '_DFT.xyz'
                if elements is not None and coordinates is not None:
                    write_xyz(os.path.join(self.path_to_workflow, xyz_filename), elements, coordinates)
                elif elements is None or coordinates is None:
                    print('Error writing xyz for: ', filename)
                    print('Make sure to check the geometry')

            # for property in properties.keys():
            dictionary_for_properties[os.path.basename(os.path.normpath(metal_ligand_complex[:-4]))] = properties

        new_descriptor_df = dataframe_from_dictionary(dictionary_for_properties)

        if printout:
            print(new_descriptor_df.to_markdown())

        if self.descriptor_df is None:
            self.descriptor_df = new_descriptor_df
        else:
            self.descriptor_df = self._merge_descriptor_dfs(self.descriptor_df, new_descriptor_df)

    def calculate_morfeus_descriptors_ce(self, metal_adduct='pristine', filename = 'filename1', solvent=None, plot_steric_map=False,printout = False, ce_name = None):
        if self.ce is not None:
            self.ce.sort()
            dictionary_for_conformer_properties = {}
            conformer_properties = {}
            print(f'Number of conformers in ensemble for: {len(self.ce)}')
            # define indexing columns that need to be excluded from boltzmann averaging
            columns_to_exclude = [f"index_{self.central_atom}", "index_donor_max", "index_donor_min",
                                    f"element_{self.central_atom}", "element_donor_max",
                                    "element_donor_min"]
            if metal_adduct.lower() == 'nbd':  # in this case there are some additional columns that need to be skipped
                columns_to_exclude += ['distance_pi_bond_1_element_1', 'distance_pi_bond_1_element_2',
                                        'distance_pi_bond_1_element_1_idx', 'distance_pi_bond_1_element_2_idx',
                                        'distance_pi_bond_2_element_1', 'distance_pi_bond_2_element_2',
                                        'distance_pi_bond_2_element_1_idx', 'distance_pi_bond_2_element_2_idx',
                                        'dihedral_angle_1_element_1', 'dihedral_angle_1_element_2',
                                        'dihedral_angle_1_element_3', 'dihedral_angle_1_element_4',
                                        'dihedral_angle_1_index_1', 'dihedral_angle_1_index_2',
                                        'dihedral_angle_1_index_3', 'dihedral_angle_1_index_4',
                                        'dihedral_angle_2_element_1', 'dihedral_angle_2_element_2',
                                        'dihedral_angle_2_element_3', 'dihedral_angle_2_element_4',
                                        'dihedral_angle_2_index_1', 'dihedral_angle_2_index_2',
                                        'dihedral_angle_2_index_3', 'dihedral_angle_2_index_4']

            # initialize list which will contain conformers to be removed if indexing properties are not
            # the same as the first conformer
            # remove_conformer_list = []

            big_df = None
            try:
                for conformer_idx, conformer in tqdm(enumerate(self.ce.conformers), total=len(self.ce.conformers)):
                    elements, coordinates = self.ce.elements, conformer.coordinates
                    conformer.properties, metal_idx, bidentate_max_donor_idx, bidentate_min_donor_idx = self._calculate_steric_electronic_desc_morfeus(solvent=solvent, dictionary=conformer.properties, elements=elements, coordinates=coordinates, filename=filename, metal_adduct=metal_adduct, plot_steric_map=plot_steric_map)
     

                    #conformer_df = np.transpose(dataframe_from_dictionary(conformer_properties))
                    #conformer_df.insert(0, f'Conformer index in {ce_name}', f" Conformer {conformer_idx + 1}")
                    #if big_df is None:
                        #big_df = conformer_df
                        #print(big_df)
                    #else:
                        #big_df = pd.concat([big_df, conformer_df])
                    #print(big_df)
                    
                    # these are indexing properties so we don't want them to be boltzmann averaged

                    # check if indexing column is the same as the first conformer, if not add conformer to
                    # remove_conformer_list to be removed from the conformer ensemble later
                    bidentate_min_donor_idx_same_across_all_conformers = True
                    bidentate_max_donor_idx_same_across_all_conformers = True
                    for key in [k for k in self.ce.get_properties().keys() if k in columns_to_exclude]:
                        if conformer.properties[key] != self.ce.conformers[0].properties[key]:
                            print("\n"
                                f"BE AWARE: Indexing property {key} is not the same across all conformers for conformer {conformer_idx}.")
                                # f"This conformer will be deleted for {os.path.basename(os.path.normpath(complex))}.")
                            # ToDo: if this is the case, rerun the conformer's properties with the min/max donor
                            # ToDo: indexing of the first conformer
                            if key == 'index_donor_min':
                                bidentate_min_donor_idx_same_across_all_conformers = False
                            elif key == 'index_donor_max':
                                bidentate_max_donor_idx_same_across_all_conformers = False
                    if not bidentate_min_donor_idx_same_across_all_conformers or not bidentate_max_donor_idx_same_across_all_conformers:
                        #print(f"BE AWARE: Min and/or max donor indexing properties are not the same across all conformers for {os.path.basename(os.path.normpath(complex))}.")
                        print("The min and/or max donor indexing properties will be recalculated for this conformer.")
                        print("Using the indexing properties of the first conformer...")
                        first_conformer_bidentate_min_donor_idx = self.ce.conformers[0].properties['index_donor_min']
                        first_conformer_bidentate_max_donor_idx = self.ce.conformers[0].properties['index_donor_max']
                        conformer.properties, metal_idx, bidentate_max_donor_idx, bidentate_min_donor_idx = self._calculate_steric_electronic_desc_morfeus(solvent=solvent, dictionary=conformer.properties, elements=elements, coordinates=coordinates, filename=filename, metal_adduct=metal_adduct, plot_steric_map=plot_steric_map, bidentate_max_donor_idx=first_conformer_bidentate_max_donor_idx, bidentate_min_donor_idx=first_conformer_bidentate_min_donor_idx)

                    conformer_df = np.transpose(dataframe_from_dictionary(conformer.properties))
                    conformer_df.insert(0, f'Conformer index in {ce_name}', f" Conformer {conformer_idx + 1}")
                    if big_df is None:
                        big_df = conformer_df

                    else:
                        big_df = pd.concat([big_df, conformer_df])
                    print(big_df)

                #             remove_conformer_list.append(conformer_idx)
                #
                # # get unique list of conformers to remove and delete them from the conformer ensemble
                # remove_conformer_list = list(set(remove_conformer_list))
                # for remove_conformer_idx in reversed(remove_conformer_list):
                #     del ce.conformers[remove_conformer_idx]
                # print(f"\nNumber of conformers in ensemble for {filename} after removing conformers with different indexing properties: {len(ce)}")

                # boltzmann averaging
                for key in [k for k in self.ce.get_properties().keys() if k not in columns_to_exclude]:
                    conformer_properties[f"{key}_boltzmann_average"] = self.ce.boltzmann_statistic(key)
                    conformer_properties[f"{key}_boltzmann_std"] = self.ce.boltzmann_statistic(key,
                                                                                            statistic='std')
                    conformer_properties[f"{key}_boltzmann_variance"] = self.ce.boltzmann_statistic(key,
                                                                                                statistic='var')
                    conformer_properties[f"{key}_Emin_conformer"] = self.ce.get_properties()[key][0]
            except Exception as e:
                # if something goes wrong with the molecular graph it usually means that the geometry is wrong
                #print('Error calculating Morfeus descriptors or Boltzmann averaging for: ', filename)
                print(e)
                print('Make sure to check the geometry')
            # all descriptors calculated, now we can write the boltzman statistics to the dictionary
                dictionary_for_conformer_properties[os.path.join(os.getcwd())] = conformer_properties
                #dictionary_for_conformer_properties[os.path.basename(os.path.normpath(self.ce))] = conformer_properties

            new_descriptor_df = dataframe_from_dictionary(dictionary_for_conformer_properties)
            if printout:
                print(new_descriptor_df.to_markdown())

            if self.descriptor_df is None:
                self.descriptor_df = new_descriptor_df
            else:
                self.descriptor_df = self._merge_descriptor_dfs(self.descriptor_df, new_descriptor_df)
            self.descriptor_df = pd.concat([big_df, self.descriptor_df], ignore_index=True)

if __name__ == "__main__":
    # example descriptor calculation for xyz files with NBD adduct in obelix/Workflow folder
    descriptors = Descriptors(central_atom='Rh', path_to_workflow=os.path.join(os.getcwd(), 'ezkell'), output_type='xyz')
    descriptors.calculate_morfeus_descriptors(solvent=None, printout=False, metal_adduct='nbd')
    descriptors.descriptor_df.to_csv('descriptors.csv', index=False)

    #the descriptors can also be calculated for 2 output types and merged into one dataframe as per example below
    conformer_descriptors = Descriptors(central_atom='Rh', path_to_workflow=os.path.join(os.getcwd(), 'ezkell'), output_type='crest')
    conformer_descriptors.calculate_morfeus_descriptors(solvent=None, printout=False, metal_adduct='nbd')
    conformer_descriptors.descriptor_df.to_csv('conformer_descriptors.csv', index=False)
    conformer_descriptors.set_output_type('xyz')
    conformer_descriptors.calculate_morfeus_descriptors(solvent=None, printout=False, metal_adduct='pristine')
    conformer_descriptors.descriptor_df.to_csv('conformer_descriptors', index=False)



    # import pickle
    # with open ("conformer_ensemble_dict_np.pkl", "rb") as f:
    #     ensemble_dict = pickle.load(f)
    # # ce1 = ensemble_dict['ceL1']
    # # descriptors = Descriptors(central_atom='Rh', ce = ce1)
    # # descriptors.calculate_morfeus_descriptors_ce(metal_adduct= 'nbd', ce_name = "ceL1")
    # # file_path = os.path.join("ce_descriptors", "ceL1_descriptors.csv")
    # # descriptors.descriptor_df.to_csv(file_path, index=False)
