from morfeus.conformer import ConformerEnsemble
from rdkit import Chem
import numpy as np
import csv
import morfeus.conformer
import pickle
from openbabel import openbabel
from mace import ComplexFromMol
import pandas as pd
import re
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.cm as cm
import os
from scipy.stats import f_oneway


class ConformerCrest:
    def __init__(self, mol_file, crest_folder = None, xyz_file = None, dft_structure = None):
        self.mol_file = mol_file
        self.crest_folder = crest_folder
        self.xyz_file = xyz_file
        self.dft_structure = dft_structure
        

    def get_rdkit_mol(self):
        rdkit_mol = Chem.MolFromMolFile(self.mol_file, removeHs = False)
        if rdkit_mol is None:
            raise ValueError("failed to create rdkit mol object")
        return rdkit_mol

    def get_connectivity_matrix(self):
        if self.get_rdkit_mol is not None:
            rdkit_mol = self.get_rdkit_mol()
            num_atoms = rdkit_mol.GetNumAtoms()
            connectivity_matrix = [[0] * num_atoms for _ in range(num_atoms)]

            for bond in rdkit_mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                order = bond.GetBondTypeAsDouble()
                connectivity_matrix[i][j] = connectivity_matrix[j][i] = order
        if connectivity_matrix is None:
            raise ValueError("failed to create connectivity matrix")
        return connectivity_matrix
    
    def get_ce(self, prune_rmsd, prune_enantiomers, prune_energy):
        ce = ConformerEnsemble.from_crest(self.crest_folder)
        ce.sort()
        ce.connectivity_matrix = self.get_connectivity_matrix()
        ce.get_energies()
        ce.generate_mol()
        if prune_enantiomers:
            ce.prune_enantiomers(keep="original")
        if prune_rmsd:
            ce.prune_rmsd()
        if prune_energy:
            ce.prune_energy()
        if ce is None:
            raise ValueError("no conformer ensemble object")
        return ce
    
    def get_absolute_energies(self, write_csv = False, csv_filepath = None):
        ce = self.get_ce(False, True, False)
        rel_en = np.array(ce.get_energies())
        with open(self.xyz_file, "r") as f:
            lines = f.readlines()
        initial_en = float(lines[1])
        abs_en = rel_en + initial_en

        #write csv file from it
        if write_csv:
            with open(csv_filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Energy'])  
                for energy in abs_en:
                    writer.writerow([energy])        
        return abs_en
    
    def get_dft_ce(self):
        elements, coordinates = morfeus.read_xyz(self.dft_structure)
        ce = ConformerEnsemble(elements, coordinates)
        ce.connectivity_matrix = self.get_connectivity_matrix()
        ce.generate_mol()
        return ce
    


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
    
class CeAnalysis:
    """
    class to analyse xTB (CREST) and DFT results
    """
    def __init__(self, dft_data = None, ce = None, crest_data = None, ce_descriptors = None, kept_conformers = None, pruned_data = None, df_folder = None, dft_ce = None, dft_descriptors = None):
        self.dft_data = dft_data
        self.ce = ce
        self.crest_data = crest_data
        self.ce_descriptors = ce_descriptors
        self.kept_conformers = kept_conformers
        self.pruned_data = pruned_data
        self.df_folder = df_folder
        self.dft_ce = dft_ce
        self.dft_descriptors = dft_descriptors

    def sort_dft_data(self, write_csv = False, output_csv = None):
        """
        This function gives the DFT info files and sort it by ascending order (conformer 1- conformer highest number) and creates a column for the conformer number
        Also excluded imfreq and troubled conformers and converting the Energy from Hartree to kj/mol
        The modified df can be saved in a csv
        """
        df = pd.read_csv(self.dft_data)
        df.insert(0, 'Conformer number', '')

        for index, row in df.iterrows():
            number = re.search(r"_([0-9]+)\.", row[1]).group(1)
            df.at[index, 'Conformer number'] = number


        #sort data based on the conformer numbers: 
        df['Conformer number'] = pd.to_numeric(df['Conformer number'])
        df_sorted = df.sort_values(by='Conformer number')
        df_sorted.reset_index(drop=True, inplace=True)
        #df_sorted = df_sorted.drop(columns=df.columns[0])
 
        #imfreq and error issues check 
        indexes_to_drop = df_sorted[df_sorted['ImFreqs'] != 0].index
        df_sorted = df_sorted[df_sorted['ImFreqs'] == 0]
        
        df_sorted.reset_index(drop=True, inplace=True)
        #from hartree to kj/mol
        df_sorted['E'] = df_sorted['E'].multiply(2625.5)
        if write_csv: 
            df_sorted.to_csv(output_csv, index=False)

        return df_sorted, indexes_to_drop
    
    def dft_data_range(self):
        """
        This function determines the higest and lowest energy of the dft df as well as the energy range (in kj/mol)
        """
        data, indexes = self.sort_dft_data()
        min_energy = data['E'].min()
        max_energy = data['E'].max()
        range = max_energy - min_energy
        return min_energy, max_energy, range
    
    def crest_data_range(self):
        """
        This function determines the higest and lowest energy of the CREST df as well as the energy range (in kj/mol)
        """
        data = self.get_crest_energies()
        min_energy = data['Energy'].min()
        max_energy = data['Energy'].max()
        range = max_energy - min_energy
        return min_energy, max_energy, range
    
    def scale_dft_energies(self):
        """
        This function adds a a column to the dft df including the standard scaled values of the energies
        """
        dft_df,  indexes_to_drop = self.sort_dft_data()
        x = dft_df.iloc[:, 4].values.reshape(-1, 1)
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        dft_df['scaled_column'] = x_scaled

        return dft_df
    
    def classify_dft_energies(self):
        """
        This function classi
        """
        dft_df = self.scale_dft_energies()
        dft_df = dft_df.sort_values(by='scaled_column')

        class_counter = 1
        threshold = dft_df['scaled_column'].iloc[0]
        dft_df['class'] = None
        for index, row in dft_df.iterrows():
            if row['scaled_column'] >= threshold + 0.06:
                class_counter += 1
                threshold = row['scaled_column']
            dft_df.at[index, 'class'] = class_counter
        dft_df = dft_df.sort_index().reset_index(drop=True)
        dft_df = dft_df.sort_index()
        return dft_df
    
    def exclude_imfreq(self, data):
        dft, indexes_to_drop = self.sort_dft_data()
        data.insert(0, "Conformer number", data.index + 1)
        data = data.drop(index=indexes_to_drop)
        data.reset_index(drop=True, inplace=True)

        return data

    def get_crest_energies(self):
        crest_df = pd.read_csv(self.crest_data)
        crest_df = self.exclude_imfreq(crest_df)
        crest_df['Energy'] = crest_df['Energy'].multiply(2625.5)
        return crest_df
    
    def get_pruned_energies(self):
        pruned_df = pd.read_csv(self.pruned_data)
        pruned_df['Energy'] = pruned_df['Energy'].multiply(2625.5)
        crest_df = self.get_crest_energies()
        pruned_df = pd.merge(pruned_df, crest_df, how='inner', on='Energy')
        pruned_df = pruned_df[['Conformer number', 'Energy']]

        return pruned_df
    
    def get_descriptors(self):
        descriptors_df = pd.read_csv(self.ce_descriptors)
        descriptors_df.drop(descriptors_df.index[-1], axis=0, inplace=True)
        descriptors_df = self.exclude_imfreq(descriptors_df)

        return descriptors_df
    
    def get_rmsd_first_conformer(self, dft = False):
        self.ce.sort()
        rmsd_matrix = self.ce.get_rmsd()
        if dft:
            rmsd_matrix = self.dft_ce.get_rmsd()
        rmsd_to_first_conformer = rmsd_matrix[0, :]
        rmsd_df = pd.DataFrame(rmsd_to_first_conformer, columns=['RMSD'])
        if self.dft_data is not None:
            rmsd_df = self.exclude_imfreq(rmsd_df)
        else: 
            rmsd_df.insert(0, "Conformer number", rmsd_df.index + 1)
        return  rmsd_df
    
    def get_rmsd_range(self, do_dft = False):
        if do_dft: 
            data = self.get_rmsd_first_conformer(dft = True)
        else: 
            data = self.get_rmsd_first_conformer()

        min_rmsd = data['RMSD'].min()
        max_rmsd = data['RMSD'].max()
        range = max_rmsd - min_rmsd
        return min_rmsd, max_rmsd, range
    
    def get_rel_energies(self):
        #in kj/mol
        energies = self.ce.get_energies()
        energies_df = pd.DataFrame(energies, columns=['Relative energy'])
        if self.dft_data is not None:
            energies_df = self.exclude_imfreq(energies_df)
        else:
            energies_df.insert(0, "Conformer number", energies_df.index + 1)
        energies_df['Relative energy'] = energies_df['Relative energy'].multiply(2625.5)
        return energies_df
    
    def get_dft_rel_energies(self):
        #in kj/mol
        dft_data = self.classify_dft_energies()
        dft_data['Relative energy'] = dft_data['E'] - dft_data['E'].iloc[0]
        return dft_data
    
    def plot_rmsd_energies (self, do_dft_rmsd = False, do_dft_energy = False, save = False, save_path = None):
        rmsd_df = self.get_rmsd_first_conformer()
        if do_dft_rmsd:
            rmsd_df = self.get_rmsd_first_conformer(dft = True)
        energies_df = self.get_rel_energies()
        if do_dft_energy:
            energies_df = self.get_dft_rel_energies()

        rmsd = rmsd_df['RMSD']
        energies = energies_df['Relative energy']
        conf_numbers = rmsd_df['Conformer number']

        for i in range(len(rmsd)):
            plt.scatter(rmsd[i], energies[i], color='darkcyan')  
            plt.text(rmsd[i], energies[i], str(conf_numbers[i]), ha='center', va='bottom', fontsize=10)
        plt.grid()
        if do_dft_rmsd:
            plt.xlabel('DFT conformers RMSD to First Conformer')
        else: 
            plt.xlabel('GNF2-xTB conformers RMSD to First Conformer')
        
        if do_dft_energy:
            plt.ylabel('Relative DFT Energy to First Conformer [kJ/mol]')
        else: 
            plt.ylabel('Relative GNF2-xTB Energy to First Conformer [kJ/mol]')
        if save: 
            plt.savefig(save_path)
        plt.show()

    def plot_dft_classes (self, relative = False):
        if relative:
            crest_df = self.get_rel_energies()
            dft_df = self.get_dft_rel_energies()
            y = crest_df['Relative energy']
            x = dft_df['Relative energy']
        else: 
            crest_df = self.get_crest_energies()
            dft_df = self.classify_dft_energies()

            y = crest_df['Energy']
            x = dft_df['E']
        dft_df2 = dft_df = self.classify_dft_energies()
        classes = dft_df2['class']
        conf_numbers = crest_df['Conformer number']
        
        unique_classes = classes.unique()
        num_classes = len(unique_classes)
        colors = plt.cm.tab20(np.linspace(0, 1, num_classes))
        color_map = dict(zip(unique_classes, colors))

        for i in range(len(x)):
            class_value = classes.iloc[i]
            color = color_map[class_value]
            plt.scatter(x[i], y[i], color=color)  
            plt.text(x[i], y[i], str(conf_numbers[i]), ha='center', va='bottom', fontsize=10)
        if relative: 
            plt.xlabel('DFT energy relative to conformer 1 [kJ/mol]')
            plt.ylabel('GFN2-xTB energy relative to conformer 1 [kJ/mol]')
        
        else:
            plt.xlabel('DFT energy [kJ/mol]')
            plt.ylabel('GFN2-xTB energy [kJ/mol]')

        plt.grid()
        plt.show()

    def plot_dft_crest_energies(self, add_kept_conformers = False, relative = False, save = False, save_path = None):
        if relative:
            crest_df = self.get_rel_energies()
            dft_df = self.get_dft_rel_energies()
            y = crest_df['Relative energy']
            x = dft_df['Relative energy']
        else:
            crest_df = self.get_crest_energies()
            dft_df = self.classify_dft_energies()

            y = crest_df['Energy']
            x = dft_df['E']
        conf_numbers = crest_df['Conformer number']

        for i in range(len(x)):
            color = 'darkcyan'
            if add_kept_conformers:
                if np.isin(i, self.kept_conformers):
                    color = "red"
            plt.scatter(x[i], y[i], color=color)  
            plt.text(x[i], y[i], str(conf_numbers[i]), ha='center', va='bottom', fontsize=10)

        if relative: 
            plt.xlabel('DFT energy relative to conformer 1 [kJ/mol]')
            plt.ylabel('GFN2-xTB energy relative to conformer 1 [kJ/mol]')
        else: 
            plt.xlabel('DFT energy [kJ/mol]')
            plt.ylabel('GFN2-xTB energy [kJ/mol]')
        plt.grid()
        if save: 
            plt.savefig(save_path)
        plt.show()

    def cluster_DBSCAN(self, eps = 0.22, plotting = False):
        rmsd_df = self.get_rmsd_first_conformer()
        energies_df = self.get_rel_energies()

        rmsd = rmsd_df['RMSD']
        energies = energies_df['Relative energy']
        conf_numbers = energies_df['Conformer number']

        X = np.column_stack((rmsd, energies))

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        dbscan = DBSCAN(eps, min_samples=2)
        cluster_labels = dbscan.fit_predict(X_scaled)
        unique_labels = np.unique(cluster_labels)
        cluster_centroids = []
        for label in unique_labels:
            if label == -1:  # Noise points
                continue
            cluster_points = X_scaled[cluster_labels == label]
            cluster_centroid = np.mean(cluster_points, axis=0)
            cluster_centroids.append(cluster_centroid)

        cluster_centroids = np.array(cluster_centroids)

        if plotting:
            plt.figure(figsize=(8, 6))
            for label in unique_labels:
                if label == -1: 
                    cluster_points = X_scaled[cluster_labels == label]
                    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c='gray', alpha=0.5, label='Noise')
                else:
                    cluster_points = X_scaled[cluster_labels == label]
                    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label}')

            plt.scatter(cluster_centroids[:, 0], cluster_centroids[:, 1], c='black', marker='x', s=100, label='Centroids')

            for conf_number, (x, y) in zip(conf_numbers, X_scaled):
                plt.annotate(conf_number, (x, y), textcoords="offset points", xytext=(0, 5), ha='center')

            plt.xlabel('RMSD to First Conformer')
            plt.ylabel('Relative GNF2-xTB Energy to First Conformer [kJ/mol]')
            #plt.title('DBSCAN Clustering with Centroids')
            plt.legend()
            plt.grid()
            plt.show()

        return X_scaled, cluster_centroids, cluster_labels 
    
    def get_cone_angle(self):
        #this gives the indexes of the dataset, does not correspond to the conformer number!!!
        descriptors = self.get_descriptors()
        min_conf = descriptors[descriptors['cone_angle'] == descriptors['cone_angle'].min()].index[0] 
        max_conf = descriptors[descriptors['cone_angle'] == descriptors['cone_angle'].max()].index[0] 

        #min_conf_number = descriptors.loc[min_conf_index]['Conformer number']
        #max_conf_number = descriptors.loc[max_conf_index]['Conformer number']


        return min_conf, max_conf
    
    def get_buried_volume(self):
        descriptors = self.get_descriptors()
        min_conf = descriptors[descriptors['buried_volume_Rh_4A'] == descriptors['buried_volume_Rh_4A'].min()].index[0] 
        max_conf = descriptors[descriptors['buried_volume_Rh_4A'] == descriptors['buried_volume_Rh_4A'].max()].index[0] 

        return min_conf, max_conf

    def get_kept_conformers(self, clustering = False, pruning = False, cone_angle = False, buried_volume = False, energy = False, eps = None, dropping_percentage = None):
        #this gives the conformer index not correspond to conformer numbers!!!
        self.kept_conformers = []
        if clustering:
            X, centroids, cluster_labels = self.cluster_DBSCAN(eps=eps)

            closest_points_indexes = []
            noise_points_indexes = []

            for centroid in centroids:
                distances = np.linalg.norm(X - centroid, axis=1)
                closest_point_index = np.argmin(distances)
                closest_point_index = closest_point_index 
                closest_points_indexes.append(closest_point_index)

            for i, label in enumerate(cluster_labels):
                if label == -1:  
                    noise_points_indexes.append(i)
                    
            self.kept_conformers = np.concatenate((closest_points_indexes, noise_points_indexes))

        if pruning:
            pruned_df = self.get_pruned_energies()
            crest_df = self.get_crest_energies()
            y = crest_df['Energy']
            indexes = []
            for i in range(len(y)):
                if y[i] in pruned_df.values:
                    indexes.append(i)
            self.kept_conformers = indexes

        if energy:
            crest_df = self.get_crest_energies()
            crest_df = crest_df.sort_values(by='Energy', ascending=False)
            num_rows = int(len(crest_df) * dropping_percentage)
            dropping_indexes = np.array(crest_df.index[:num_rows])
            filtered_kept_conformers = [conformer for conformer in self.kept_conformers if conformer not in dropping_indexes]
            self.kept_conformers = filtered_kept_conformers


        if cone_angle:
            min, max = self.get_cone_angle()
            self.kept_conformers.append(min)
            self.kept_conformers.append(max)


        if buried_volume:
            min, max = self.get_buried_volume()
            self.kept_conformers.append(min)
            self.kept_conformers.append(max)


        self.kept_conformers = np.unique(self.kept_conformers)
        return self.kept_conformers
    
    def get_dropped_conformers(self):
        #total_conformers = np.arange(1, len(self.get_crest_energies()))
        dft_data = self.classify_dft_energies()
        total_conformers = dft_data["Conformer number"]
        dropped_conformers = np.setdiff1d(total_conformers, self.kept_conformers)
        return dropped_conformers
    
    def get_confusion_matrix(self):
        kept_conformers = self.kept_conformers
        dropped_conformers = self.get_dropped_conformers()
        dft_df = self.classify_dft_energies()

        classes = dft_df['class'].unique()
        true_positive = 0
        false_negative = 0
        for i in classes:
            class_indexes = dft_df.index[dft_df['class'] == i]
            index_in_array = np.any(np.isin(class_indexes, kept_conformers))
            if index_in_array:
                true_positive = true_positive + 1
            else:
                false_negative = false_negative + 1

        false_positive = len(kept_conformers) - true_positive
        true_negative = len(kept_conformers) + len(dropped_conformers) - sum([true_positive, false_negative, false_positive])


        return true_positive, false_negative, false_positive, true_negative
    
    def missed_classes(self, plotting = False, relative = False):
        kept_conformers = self.kept_conformers
        dft_df = self.classify_dft_energies()
        classes = dft_df['class'].unique()
        missed_class = []

        for i in classes:
            class_indexes = dft_df.index[dft_df['class'] == i]
            index_in_array = np.any(np.isin(class_indexes, kept_conformers))
            if index_in_array == False:
                missed_class.append(i)
    
        data = self.classify_dft_energies()
        missed = data.index[data['class'].isin(missed_class)].tolist()
        print("what is going on")
        #plotting:
        if plotting:
            if relative: 
                crest_df = self.get_rel_energies()
                dft_df = self.get_dft_rel_energies()
                y = crest_df['Relative energy']
                x = dft_df['Relative energy']
            else:
                crest_df = self.get_crest_energies()
                y = crest_df['Energy']
                x = dft_df['E']
            conf_numbers = crest_df['Conformer number']


            for i in range(len(x)):
                color = 'darkcyan'
                if np.isin(i, missed):
                    color = "gold"
                plt.scatter(x[i], y[i], color=color)  
                plt.text(x[i], y[i], str(conf_numbers[i]), ha='center', va='bottom', fontsize=10)
            if relative: 
                plt.xlabel('DFT relative energy to first conformer [kJ/mol]')
                plt.ylabel('GFN2-xTB relative energy to first conformer [kJ/mol]')
            else: 
                plt.xlabel('DFT energy [kJ/mol]')
                plt.ylabel('GFN2-xTB energy [kJ/mol]')
            plt.grid()
            plt.show()
    
        return missed

    def evaluate_dataframes(self, exclude_ceL172 = False, plotting2d = False, plotting1d = False, ratio = False):
        eval_df = pd.DataFrame(columns=['method', 'missed key conf', 'dropped conf'])
        for file in os.listdir(self.df_folder):
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(self.df_folder, file))
                method_name = os.path.splitext(file)[0]
                if exclude_ceL172:
                    df = df[~df['ce number'].str.contains('ceL172')]

                missed_sum = df['false negative'].sum()
                dropped_sum = df['true negative'].sum()
                if ratio:
                    rat = dropped_sum / missed_sum
                    eval_df = eval_df.append({'method': method_name, 'missed key conf': missed_sum, 'dropped conf': dropped_sum, 'ratio': rat}, ignore_index=True)
                else:
                    eval_df = eval_df.append({'method': method_name, 'missed key conf': missed_sum, 'dropped conf': dropped_sum}, ignore_index=True)

        if plotting2d:
            plt.figure(figsize=(10, 6))
            for method in eval_df['method'].unique():
                method_df = eval_df[eval_df['method'] == method]
                plt.scatter( method_df['missed key conf'], method_df['dropped conf'], label=method)

            plt.ylabel('Number of dropped conformers')
            plt.xlabel('Number of missed key conformers')
            plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.subplots_adjust(right=0.7)
            plt.grid()
            plt.show()

        if plotting1d:
            plt.figure(figsize=(10, 6))
            for i, method in enumerate(eval_df['method'].unique()):
                method_df = eval_df[eval_df['method'] == method]
                plt.plot([i] * len(method_df), method_df['dropped conf'],  'o', label=method)

            
            plt.ylabel('Number of dropped conformers')
            plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.subplots_adjust(right=0.7)
            plt.grid()
            plt.gca().set_xticklabels([])
            #plt.xticks(range(len(eval_df['method'].unique())), eval_df['method'].unique(), rotation=45)
            plt.show()
        return eval_df
    
    def predict_conformers(self):
        energy_df = self.get_rel_energies()
        selected_values = energy_df.loc[self.kept_conformers, 'Conformer number']
        return selected_values
    
    def select_descriptors(self):
        all_steric_params = ['NE_quad', 'NW_quad', 'SW_quad',
                     'SE_quad', '+,+,+_octant', '-,+,+_octant', '-,-,+_octant',
                     '+,-,+_octant', '+,-,-_octant', '-,-,-_octant', '-,+,-_octant',
                     '+,+,-_octant', 'buried_volume_Rh_3.5A', 'buried_volume_donor_max',
                     'buried_volume_donor_min', 'buried_volume_Rh_4A', 'buried_volume_Rh_5A',
                     'buried_volume_Rh_6A', 'buried_volume_Rh_7A', 'min_bv_donor', 'max_bv_donor', 'std_quad',
                     'min_quad', 'max_quad', 'std_oct', 'min_oct', 'max_oct',
                     'ratio_std_oct_std_quad', 'ratio_min_oct_min_quad', ]
        all_geometric_params = ['dihedral_angle_1', 'dihedral_angle_2', 'bite_angle', 'cone_angle',
                        'distance_Rh_max_donor_gaussian', 'distance_Rh_min_donor_gaussian', 'sasa_gfn2_xtb',
                        'Rh_donor_min_d', 'Rh_donor_max_d', 'bite_angle_sin', 'bite_angle_cos',
                        'cone_angle_sin', 'cone_angle_cos']
        free_ligand_params = ['free_ligand_dipole_moment_dft', 'free_ligand_lone_pair_occupancy_min_donor_dft',
                      'free_ligand_lone_pair_occupancy_max_donor_dft',
                      'free_ligand_dispersion_energy_dft', 'free_ligand_nbo_charge_min_donor_dft',
                      'free_ligand_nbo_charge_max_donor_dft', 'free_ligand_mulliken_charge_min_donor_dft',
                      'free_ligand_mulliken_charge_max_donor_dft', 'free_ligand_homo_energy_dft',
                      'free_ligand_lumo_energy_dft', 'free_ligand_homo_lumo_gap_dft', 'free_ligand_hardness_dft',
                      'free_ligand_softness_dft', 'free_ligand_electronegativity_dft',
                      'free_ligand_electrophilicity_dft']
        all_electronic_and_thermodynamic_params = ['distance_pi_bond_1', 'distance_pi_bond_2', 'dispersion_p_int_Rh_gfn2_xtb',
                                           'dispersion_p_int_donor_max_gfn2_xtb',
                                           'dispersion_p_int_donor_min_gfn2_xtb', 'ip_gfn2_xtb',
                                           'dipole_gfn2_xtb', 'ea_gfn2_xtb', 'electrofugality_gfn2_xtb',
                                           'nucleofugality_gfn2_xtb', 'nucleophilicity_gfn2_xtb',
                                           'electrophilicity_gfn2_xtb', 'HOMO_LUMO_gap_gfn2_xtb',
                                           'sum_electronic_and_free_energy_dft', 'sum_electronic_and_enthalpy_dft',
                                           'zero_point_correction_dft', 'entropy_dft', 'dipole_moment_dft',
                                           'lone_pair_occupancy_min_donor_dft',
                                           'lone_pair_occupancy_max_donor_dft', 'dispersion_energy_dft',
                                           'nbo_charge_Rh_dft', 'nbo_charge_min_donor_dft',
                                           'nbo_charge_max_donor_dft', 'mulliken_charge_Rh_dft',
                                           'mulliken_charge_min_donor_dft', 'mulliken_charge_max_donor_dft',
                                           'homo_energy_dft', 'lumo_energy_dft', 'homo_lumo_gap_dft',
                                           'hardness_dft', 'softness_dft', 'electronegativity_dft',
                                           'electrophilicity_dft', 'min_NBO_donor', 'max_NBO_donor',
                                           'lone_pair_occ_min', 'lone_pair_occ_max', 'nbo_charge_max_donor_dft_abs_diff',
                                           'nbo_charge_min_donor_dft_abs_diff',	'mulliken_charge_max_donor_dft_abs_diff',
                                           'mulliken_charge_min_donor_dft_abs_diff', 'lone_pair_occupancy_max_donor_dft_abs_diff',
                                           'lone_pair_occupancy_min_donor_dft_abs_diff'] + free_ligand_params
        all_descriptors = all_steric_params + all_geometric_params + all_electronic_and_thermodynamic_params
        return all_descriptors
    
    def filter_descriptors(self, d = False):
        all_descriptors = self.select_descriptors()
        df = self.get_descriptors()
        valid_columns = set(all_descriptors).intersection(df.columns)
        df_filtered = df[[df.columns[0]] + list(valid_columns)]
        if d:
            df = self.sort_dft_descriptors()
            df_filtered = df[[df.columns[0]] + list(valid_columns)]
        return df_filtered
    
    def sort_dft_descriptors(self):
        df = pd.read_csv(self.dft_descriptors)
        #df.insert(0, 'Conformer number', '')
        df['filename_tud'] = df['filename_tud'].str.split('_', n=2).str[-1]
        #df['Conformer number'] = df['filename_tud']
        df['filename_tud'] = pd.to_numeric(df['filename_tud'])
        df_sorted = df.sort_values(by='filename_tud')
        df_sorted.reset_index(drop=True, inplace=True)
        self.exclude_imfreq(data = df_sorted)
        return df_sorted
    
    def do_anova_test(self, crest_df, dft_df, column):
        crest_df = pd.read_csv(crest_df)
        dft_df = pd.read_csv(dft_df)
        method_1 = crest_df.iloc[:, column]
        method_2 = dft_df.iloc[:, column]
        descriptor = crest_df.columns[column]
        f_statistic, p_value = f_oneway(method_1, method_2)
        return f_statistic, p_value, descriptor

if __name__ == "__main__":
    test_crest = ConformerCrest(mol_file = os.path.join(os.getcwd(), 'tests', 'L1.mol'), crest_folder = os.path.join(os.getcwd(), 'tests', 'L1'))
    test_ob = ConformerOpenbabel(mol_file = os.path.join(os.getcwd(), 'tests', 'L1.mol'), metal_symbol= 'Rh')
    test_rdkit = ConformerRdkit(mol_file = os.path.join(os.getcwd(), 'tests', 'L1.mol'), metal_symbol= 'Rh')
    test_analysis = CeAnalysis(dft_descriptors = os.path.join(os.getcwd(), 'tests', 'DFT_descriptors_ceL7.csv'))