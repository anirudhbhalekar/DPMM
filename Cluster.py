import numpy as np 
import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal 
import csv 
import pickle
from EM import GMM
from DPMM import DPMM
import random


##############################################################################################
#
#   Author: Ani Bhalekar
#   Date: 06/08/2024 
#   
#   Purpose: 
#           Class for general clustering applications
#           EM - added on 30/07/2024
#           DPMM - added on 06/08/2024
#
##############################################################################################

DUMP_LOC = "iGMM/"

class Clusterer: 
    def __init__(self) -> None:
        self.init = True
    
    def load_csv(self, filename, label_col, d_start_col, d_end_col): 
        # Loads csv - need to know which column is the label/category and the span of the data in the csv
        self.label_arr = []
        self.data_arr = []

        with open(filename) as file: 
            reader_obj = csv.reader(file)
            for i, row in tqdm.tqdm(enumerate(reader_obj)):
                if i == 0: 
                    self.title = row[0]
                    continue 
                self.label_arr.append(row[label_col])
                self.data_arr.append(list([float(x) for x in row[d_start_col:d_end_col]]))
        
        shuff = list(zip(self.data_arr, self.label_arr))
        random.shuffle(shuff)
        self.data_arr, self.label_arr = zip(*shuff)

        self.data_arr = np.array(self.data_arr)/100
        self.leg_label_arr = self.label_arr


    def degrade_signal(self, signal, var = 1): 
        # Method to degrade signal (noise addition)
        degraded_signal = [val + np.random.normal(0, np.sqrt(var)) for val in signal]
        return degraded_signal    

    def dim_reduce(self, train_size, n_components, degradation = 0): 
        # Dimensionality reduction via PCA
        if train_size == -1: train_size = len(self.data_arr)

        da = [self.degrade_signal(sig, degradation) for sig in self.data_arr]
        pca = PCA(n_components=n_components)
        pca.fit(da)
        da = pca.transform(da)

        self.pca = pca
        self.transformed_da = da
        self.train_size = train_size
    
    def vis_3D(self, is_categorical = False): 
        # Method to visualise 3 Dimensions of PCA 
        # If the data is categorical, we seperate out component wise, otherwise a continuous colorbar is implemented

        self.dim_reduce(-1, 3)
        fig = plt.figure(1, figsize=(4, 3))
        plt.clf()
        ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)
        plt.cla()

        if is_categorical:
            label_set = sorted(list(set(self.label_arr)))
            label_ids = np.arange(len(label_set))
            loc_dict = {}
            for i in label_ids: 
                loc_dict[label_set[i]] = i 
            y = []
            for title in self.label_arr: 
                y.append(loc_dict[title])
        else: 
            y = [float(l) for l in self.label_arr]

        s_plot = ax.scatter(self.transformed_da[:, 0], self.transformed_da[:, 1], self.transformed_da[:, 2], c = y, cmap=plt.cm.jet, edgecolor="k")
        cbar = fig.colorbar(s_plot)
        cbar.set_label("Shim Mass", rotation = 270)
        plt.title("Component Representation")
        plt.show()


    def discretize_labels(self, num_categories, loc_label_seed = None): 
        # Discretize the labels if they are not already discretized

        if num_categories < 0: 
            self.label_arr = self.label_arr
            self.loc_labels = sorted(list(set(self.label_arr)))
            return 
        
        self.label_arr = [float(l) for l in self.label_arr]
        min_label, max_label = min(self.label_arr), max(self.label_arr)
        self.loc_labels = np.linspace(min_label, max_label, num_categories)

        if loc_label_seed is not None: 
            self.loc_labels = loc_label_seed
            num_categories = len(loc_label_seed)
        
        self.num_categories = len(self.loc_labels)

        for i, label in enumerate(self.label_arr): 
            norm_index = list(filter(lambda x: self.loc_labels[x] >= label, range(len(self.loc_labels))))[0]
            if norm_index >= num_categories: norm_index = num_categories - 1
            if norm_index < 0: norm_index = 0 
            self.label_arr[i] = self.loc_labels[int(norm_index)]
    
    def separate_one(self, i, labels, da):
        category = self.loc_labels[i]

        da_cat = list()
        for i, arr in enumerate(da): 
            if labels[i] == category: 
                da_cat.append(arr)
        da_cat = np.array(da_cat)

        return da_cat
    
    def EM_train(self, train_size, n_components, num_categories = 10, degradation = 0, k = 2, loc_label_seed = None):
        
        self.dim_reduce(train_size, n_components, degradation)
        self.discretize_labels(num_categories, loc_label_seed)
        dim = np.shape(self.transformed_da[0])[0]
        train_dict = {} 

        train_label, train_da = self.label_arr[:self.train_size], self.transformed_da[:self.train_size]

        for i in range(len(self.loc_labels)): 
            gmm = GMM(k, dim)
            da_cat = self.separate_one(i, train_label, train_da)
            da_cat = np.array(da_cat)
            gmm.init_em(da_cat)
            print("EM Alg Initialised for ", self.loc_labels[i])
            for j in range(200): 
                gmm.e_step()
                gmm.m_step()

            print("EM Alg Completed for " , self.loc_labels[i])
            train_dict[self.loc_labels[i]] = [gmm.mu, gmm.sigma, gmm.pi]
        
        self.train_dict = train_dict
    
    def DPMM_train(self, train_size, n_components, num_categories = 10, degradation = 0, loc_label_seed = None): 

        self.dim_reduce(train_size, n_components, degradation)
        self.discretize_labels(num_categories, loc_label_seed=loc_label_seed)
        train_dict = {} 
        dpmm_dict = {}

        train_label, train_da = self.label_arr[:self.train_size], self.transformed_da[:self.train_size]

        for i in range(len(self.loc_labels)): 
            i = len(self.loc_labels) - i - 1
            da_cat = self.separate_one(i, train_label, train_da)
            da_cat = np.array(da_cat)
            if len(da_cat) < 1: 
                continue
            dpmm = DPMM(alpha=1e-4)
            dpmm.fit(X = da_cat, n_iterations=20)
            dpmm_dict[self.loc_labels[i]] = dpmm
            train_dict[self.loc_labels[i]] = [dpmm.means, dpmm.covariances, dpmm.weights]
            print("NUMBER OF CLUSTERS :", dpmm.components)

        self.dpmm_dict = dpmm_dict
        self.train_dict = train_dict

        self.store_dpmm()

    def EM_test(self): 
        self.test_answers = []
        self.test_predictions = []
        count = 0 

        test_labels, test_data = self.label_arr[:], self.transformed_da[:]
        for loc_label, data in zip(test_labels, test_data): 
            if count % 200 == 0: 
                print("Test " + str(count) +  " of " + str(len(self.data_arr)))
            count += 1
            p_max, pred_label = -np.inf, None
            for y in self.loc_labels: 
                if not y in self.train_dict.keys(): continue
                for mu, sigma in zip(self.train_dict[y][0], self.train_dict[y][1]): 
                    this_p = multivariate_normal.logpdf(data, mean = mu, cov = sigma, allow_singular=True)
                    if this_p > p_max: 
                        p_max = this_p
                        pred_label = y
            
            self.test_predictions.append(pred_label)
            self.test_answers.append(loc_label)
    
    def DPMM_test(self): 
        try: 
            self.load_dpmm()
        except FileNotFoundError:
            print("Run DPMM Train Once")
            raise FileNotFoundError
        
        self.test_answers = []
        self.test_predictions = []
        self.prob_predictions = []
        count = 0 

        self.test_answers, test_data = self.label_arr[:], self.transformed_da[:]
        print("Starting Tests")
        all_log_probs = []
        for label in self.loc_labels: 
            print(f"DPMM Loaded for {label}")
            try:
                this_dpmm = self.dpmm_dict[label]
                log_probs = this_dpmm.return_log_probs(test_data)
                all_log_probs.append(log_probs)
            except KeyError: 
                continue
        
        self.prob_predictions = np.transpose(np.array(all_log_probs))

        for i in tqdm.tqdm(range(len(self.prob_predictions))): 
            ind = np.argmax(self.prob_predictions[count])
            self.test_predictions.append(self.loc_labels[ind])
            count += 1

    def heat_map(self): 
        on_bin_accuracy = 0
        match_array = np.zeros((len(self.loc_labels), len(self.loc_labels)))
        for tr, ta in zip(self.test_predictions, self.test_answers): 
            i_tr, i_ta = list(self.loc_labels).index(tr), list(self.loc_labels).index(ta)
            if i_tr == i_ta: 
                on_bin_accuracy += 1
            match_array[i_ta][i_tr] += 1
        
        match_array /= match_array.sum(axis = 0)
        on_bin_accuracy /= len(self.test_answers)
        print("ACCURACY ON BIN : " , 100 * on_bin_accuracy)
        fig, ax = plt.subplots()
        im = ax.imshow(match_array, vmin = 0, vmax = 1, cmap = "magma")
        
        print(match_array)
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Accuracy (given actual data)", rotation=-90, va="bottom")

        try:
            self.loc_labels = [round(t, 2) for t in self.loc_labels]
        except: 
            pass
        ax.set_xticks(np.arange(len(self.loc_labels)), labels=self.loc_labels, rotation = -90)
        ax.set_yticks(np.arange(len(self.loc_labels)), labels=self.loc_labels)

        ax.set_ylabel("Actual outupt")
        ax.set_xlabel("Predicted output")

        ax.set_title("Match between actual and predicted outputs")
        fig.tight_layout()
        plt.show()

    def store_dpmm(self): #
        dpmm_file = open("dpmm_dict", "ab")
        pickle.dump(self.dpmm_dict, dpmm_file)
        dpmm_file.close()

    def load_dpmm(self): 
        dpmm_file = open("dpmm_dict", "rb")
        self.dpmm_dict = pickle.load(dpmm_file)
        dpmm_file.close()
        
        
if __name__ == "__main__": 

    filename = "ENTER CSV ROW FILE"

    clusterer = Clusterer()
    clusterer.load_csv(f"{filename}.csv", 0, 2, 45)
    clusterer.DPMM_train(train_size=600, n_components=10, num_categories=-1, degradation=0)
    clusterer.DPMM_test()
    clusterer.heat_map()
