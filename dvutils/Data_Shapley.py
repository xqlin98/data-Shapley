from tqdm import tqdm
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import itertools
import math

class Data_Shapley():
    
    def __init__(self, train_loader, val_loader, model_train):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_train = model_train

    def get_subset_loader(self, selected_idx):
        return DataLoader(self.train_loader.dataset, sampler=SubsetRandomSampler(selected_idx), batch_size=self.model_train.batch_size)

    def mc_one_iteration(self, metric, permutation=None):
        """Runs one iteration of Monte Carlo-Shapley algorithm."""
        marginal_contribs = np.zeros([self.n_dp])

        new_score = self.get_null_score(metric)
        selected_idx = []
        for n, idx in tqdm(enumerate(permutation), desc="One Monte Carlo iteration", position=1, leave=False):
            old_score = new_score
            selected_idx.append(idx)
            tmp_loader = self.get_subset_loader(selected_idx)
            
            self.model_train.fit(tmp_loader, self.val_loader)
            loss, accuracy = self.model_train.evaluate(self.val_loader)
            
            new_score = -loss if metric == "neg_loss" else accuracy
            marginal_contribs[idx] = (new_score - old_score)
            
            self.restart_model()
        return marginal_contribs


    def mc_one_iteration_idx(self, idx, metric, permutation=None):
        """Runs one iteration of Monte Carlo-Shapley algorithm."""
        perm_idx = np.where(permutation==idx)[0].item()

        selected_idx = permutation[:perm_idx]
        if perm_idx == 0:
            old_score = self.get_null_score(metric)
        else:
            tmp_loader = self.get_subset_loader(selected_idx)
            self.model_train.fit(tmp_loader, self.val_loader)
            loss, accuracy = self.model_train.evaluate(self.val_loader)
            old_score = -loss if metric == "neg_loss" else accuracy
            self.restart_model()
        
        selected_idx = np.append(selected_idx, idx)
        tmp_loader = self.get_subset_loader(selected_idx)
        self.model_train.fit(tmp_loader, self.val_loader)
        loss, accuracy = self.model_train.evaluate(self.val_loader)
        new_score = -loss if metric == "neg_loss" else accuracy
        self.restart_model()
        return new_score - old_score
    
    def get_null_score(self, metric):
        """To compute the performance with initial weight"""
        try:
            self.null_score
        except:
            self.restart_model()
            loss, accuracy = self.model_train.evaluate(self.val_loader)
            self.null_score = -loss if metric == "neg_loss" else accuracy
        return self.null_score

    def restart_model(self):
      self.model_train.restart_model()

    def run_all(self,
            method="mc",
            mc_iteration=2000,  
            metric="neg_loss"):
        """Runs the Monte Carlo-Shapley algorithm."""
        self.n_dp = len(self.train_loader.dataset)
        
        if method == "mc":
            sv_result = np.zeros([self.n_dp])
            for _ in tqdm(range(mc_iteration), desc='[Running Monte Carlo for all data points]',position=0):
                permutation = np.random.permutation(self.n_dp)
                marginal_contribs = self.mc_one_iteration(metric, permutation)
                sv_result += marginal_contribs/mc_iteration
        elif method == "exact":
            # iterate over all permutations to get the exact Shapley values
            sv_result = np.zeros([self.n_dp])
            all_perm = itertools.permutations(range(self.n_dp))
            num_perm = math.factorial(self.n_dp)
            for permutation in tqdm(all_perm, desc='[Exact Shapley]'):
                marginal_contribs = self.mc_one_iteration(metric, permutation)
                sv_result += marginal_contribs/num_perm
        return sv_result
    

    def run_idx(self,
            idx,
            method="mc",
            mc_iteration=2000,  
            metric="neg_loss"):
        """Runs the Monte Carlo-Shapley algorithm."""
        self.n_dp = len(self.train_loader.dataset)
        
        if method == "mc":
            sv_result = 0
            for _ in tqdm(range(mc_iteration), desc='[Runing Monte Carlo for one data point]'):
                permutation = np.random.permutation(self.n_dp)
                marginal_contribs = self.mc_one_iteration_idx(idx, metric, permutation)
                sv_result += marginal_contribs/mc_iteration
        elif method == "exact":
            # iterate over all permutations to get the exact Shapley values
            sv_result = 0
            all_perm = itertools.permutations(range(self.n_dp))
            num_perm = math.factorial(self.n_dp)
            for permutation in tqdm(all_perm, desc='[Runing exact Shapley for one data point]'):
                marginal_contribs = self.mc_one_iteration_idx(idx, metric, np.array(permutation))
                print(marginal_contribs)
                sv_result += marginal_contribs/num_perm
        return sv_result