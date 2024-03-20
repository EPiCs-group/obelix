#this code was a lotta work, im quite proud of it

import pandas as pd
import re
import matplotlib.pyplot as plt
import pickle
from morfeus.conformer import ConformerEnsemble
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.cm as cm

class CeAnalysis:
    def __init__(self, dft_data = None, ce = None, crest_data = None, ce_descriptors = None, kept_conformers = None, pruned_data = None):
        self.dft_data = dft_data
        self.ce = ce
        self.crest_data = crest_data
        self.ce_descriptors = ce_descriptors
        self.kept_conformers = kept_conformers
        self.pruned_data = pruned_data

    def sort_dft_data(self, write_csv = False, output_csv = None):
        df = pd.read_csv(self.dft_data)
        df.insert(0, 'Conformer number', '')

        for index, row in df.iterrows():
            number = re.search(r"_([0-9]+)\.", row[1]).group(1)
            df.at[index, 'Conformer number'] = number


        #sort data based on the conformer numbers: 
        df['Conformer number'] = pd.to_numeric(df['Conformer number'])
        df_sorted = df.sort_values(by='Conformer number')
        df_sorted.reset_index(drop=True, inplace=True)
        df_sorted = df_sorted.drop(columns=df.columns[0])

        if write_csv: 
            df_sorted.to_csv(output_csv, index=False) 

        return df_sorted
    
    def scale_dft_energies(self):
        dft_df = self.sort_dft_data()
        x = dft_df.iloc[:, 3].values.reshape(-1, 1)
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        dft_df['scaled_column'] = x_scaled

        return dft_df
    
    def classify_dft_energies(self):
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

        return dft_df
    
    def get_crest_energies(self):
        crest_df = pd.read_csv(self.crest_data)

        return crest_df
    
    def get_pruned_energies(self):
        pruned_df = pd.read_csv(self.pruned_data)

        return pruned_df
    
    def get_descriptors(self):
        descriptors = pd.read_csv(self.ce_descriptors)
        #self.ce_descriptors = descriptors

        return descriptors
    
    def get_rmsd_first_conformer(self):
        self.ce.sort()
        rmsd_matrix = self.ce.get_rmsd()
        rmsd_to_first_conformer = rmsd_matrix[0, :]
        return  rmsd_to_first_conformer
    
    def plot_rmsd_energies (self):
        rmsd = self.get_rmsd_first_conformer()
        energies = self.ce.get_energies()

        for i in range(len(rmsd)):
            plt.scatter(rmsd[i], energies[i], color='darkcyan')  
            plt.text(rmsd[i], energies[i], str(i+1), ha='center', va='bottom', fontsize=10)
        plt.grid()
        plt.xlabel('RMSD to First Conformer')
        plt.ylabel('Relative xTB Energy to First Conformer')
        plt.show()

    def plot_dft_classes (self):
        crest_df = self.get_crest_energies()
        dft_df = self.classify_dft_energies()

        y = crest_df.iloc[:, 0]
        x = dft_df.iloc[:, 3]
        classes = dft_df['class']
        
        unique_classes = classes.unique()
        num_classes = len(unique_classes)
        colors = plt.cm.tab20(np.linspace(0, 1, num_classes))
        color_map = dict(zip(unique_classes, colors))

        for i in range(len(x)):
            class_value = classes.iloc[i]
            color = color_map[class_value]
            plt.scatter(x[i], y[i], color=color)  
            plt.text(x[i], y[i], str(i+1), ha='center', va='bottom', fontsize=10)

        plt.xlabel('DFT energy [hartree]')
        plt.ylabel('GFN2-xTB energy [hartree]')
        plt.grid()
        plt.show()

    def plot_dft_crest_energies(self, add_kept_conformers = False):
        crest_df = self.get_crest_energies()
        dft_df = self.classify_dft_energies()

        y = crest_df.iloc[:, 0]
        x = dft_df.iloc[:, 3]
        
        for i in range(len(x)):
            color = 'darkcyan'
            if add_kept_conformers:
                if np.isin(i+1, self.kept_conformers):
                    color = "red"
            plt.scatter(x[i], y[i], color=color)  
            plt.text(x[i], y[i], str(i+1), ha='center', va='bottom', fontsize=10)


        plt.xlabel('DFT energy [hartree]')
        plt.ylabel('GFN2-xTB energy [hartree]')
        plt.grid()
        plt.show()

    def cluster_DBSCAN(self, eps = 0.3, plotting = False):
        rmsd = self.get_rmsd_first_conformer()
        energies = self.ce.get_energies()
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

            for i, txt in enumerate(range(1, X.shape[0] + 1)):
                plt.annotate(txt, (X_scaled[i, 0], X_scaled[i, 1]), textcoords="offset points", xytext=(0, 5), ha='center')

            plt.xlabel('RMSD to First Conformer')
            plt.ylabel('Energies to First Conformer')
            plt.title('DBSCAN Clustering with Centroids')
            plt.legend()
            plt.grid()
            plt.show()

        return X_scaled, cluster_centroids, cluster_labels 
    
    def get_cone_angle(self):
        #this gives the conformer index starting from 1 (not 0!!)
        descriptors = self.get_descriptors()
        min_conf = descriptors[descriptors['cone_angle'] == descriptors['cone_angle'].min()].index[0] + 1
        max_conf = descriptors[descriptors['cone_angle'] == descriptors['cone_angle'].max()].index[0] + 1

        return min_conf, max_conf
    
    def get_buried_volume(self):
        #this gives the conformer index starting from 1 (not 0!!)
        descriptors = self.get_descriptors()
        min_conf = descriptors[descriptors['buried_volume_Rh_4A'] == descriptors['buried_volume_Rh_4A'].min()].index[0] + 1
        max_conf = descriptors[descriptors['buried_volume_Rh_4A'] == descriptors['buried_volume_Rh_4A'].max()].index[0] + 1

        return min_conf, max_conf

    def get_kept_conformers(self, clustering = False, pruning = False, cone_angle = False, buried_volume = False):
        #this gives the conformer index starting from 1 (not 0!!)
        self.kept_conformers = []
        if clustering:
            X, centroids, cluster_labels = self.cluster_DBSCAN()

            closest_points_indexes = []
            noise_points_indexes = []

            for centroid in centroids:
                distances = np.linalg.norm(X - centroid, axis=1)
                closest_point_index = np.argmin(distances)
                closest_point_index = closest_point_index + 1
                closest_points_indexes.append(closest_point_index)

            for i, label in enumerate(cluster_labels):
                if label == -1:  
                    noise_points_indexes.append(i + 1)
                    
            self.kept_conformers = np.concatenate((closest_points_indexes, noise_points_indexes))

        if pruning:
            pruned_df = self.get_pruned_energies()
            crest_df = self.get_crest_energies()
            y = crest_df.iloc[:, 0]
            indexes = []
            for i in range(len(y)):
                if y[i] in pruned_df.values:
                    indexes.append(i+1)
            self.kept_conformers = indexes
        
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
        total_conformers = np.arange(1, len(self.get_crest_energies()) + 1)
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
            index_in_array = np.any(np.isin(class_indexes, kept_conformers -1))
            if index_in_array:
                true_positive = true_positive + 1
            else:
                false_negative = false_negative + 1

        false_positive = len(kept_conformers) - true_positive
        true_negative = len(kept_conformers) + len(dropped_conformers) - sum([true_positive, false_negative, false_positive])


        return true_positive, false_negative, false_positive, true_negative



if __name__ == "__main__":
    with open("conformer_ensemble_dict_np.pkl", "rb") as f:
        ensemble_dict = pickle.load(f)
    ce = ensemble_dict["ceL17"]
    check = CeAnalysis(dft_data = f"C:\\Users\\finta\\Documents\\tu_delft\\MEP\\DFT_input\\DFT_output\\ce71\\info_ceL71.csv", crest_data= 
                       f"C:\\Users\\finta\\Documents\\tu_delft\\MEP\\CREST\\Absolute_energies\\abs_en_ceL71.csv",
                       ce = ce, pruned_data= f"C:\\Users\\finta\\Documents\\tu_delft\\MEP\\CREST\\Absolute_energies\\rmsd_pruning\\abs_en_rp_ceL71.csv",
                       ce_descriptors= f"C:\\Users\\finta\\Documents\\tu_delft\\MEP\\CREST\\ce_descriptors\\descriptor_df_crest_ceL71.csv")
    #dft_data = check.sort_dft_data()
    #dft_data = check.classify_dft_energies()
    #crest_data = check.get_crest_energies()
    #rmsd = check.get_rmsd_first_conformer()
    #check.plot_rmsd_energies()
    #check.plot_dft_classes()
    #check.plot_dft_crest_energies()
    #X_scaled, cluster_centroids, cluster_labels = check.cluster_DBSCAN(plotting=True)
    check.kept_conformers = check.get_kept_conformers(cone_angle=False, buried_volume= False, pruning=False, clustering=True)
    #dropped_conformers = check.get_dropped_conformers()
    check.plot_dft_crest_energies(add_kept_conformers=True)
    tp, fn, fp, tn = check.get_confusion_matrix()
    print(tp, fn, fp, tn)
    



    





    
