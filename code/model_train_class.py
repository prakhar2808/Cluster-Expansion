from structure_helper_class import structure_helper
from matrix_inversion_class import matrix_inversion

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from tabulate import tabulate

class model_train:
    
    #Constructor.
    def __init__(self, structure_name_to_object_map):
        self.training_dataset_structures_list_ = self.generate_structures_for_training_data(structure_name_to_object_map)
        self.testing_dataset_structures_list_ = self.generate_structures_for_testing_data(structure_name_to_object_map)
        self.X_training_data_, self.Y_training_data_ = self.generate_data_vectors('train')
        self.X_testing_data_, self.Y_testing_data_ = self.generate_data_vectors('test')
        self.lasso_object_ = self.get_lasso_object(structure_name_to_object_map)
        self.lr_object_ = self.get_lr_object(structure_name_to_object_map)
        self.matinv_object_ = self.get_matinv_object(structure_name_to_object_map)
    
    #Generates a list of structures for training.
    def generate_structures_for_training_data(self, structure_name_to_object_map, display_training_structs = False):
        #Getting compositions
        composition_ratio_to_structure_names_list_map = structure_helper.get_composition_ratio_to_structure_names_list_map(structure_name_to_object_map.values())
        #Generating structures for training data
        training_dataset_structures_list = []
        for composition,structures_names_list in composition_ratio_to_structure_names_list_map.items():
            structures_list = []
            for structure_name in structures_names_list:
                structures_list.append(structure_name_to_object_map[structure_name])
            training_dataset_structures_list.append(structure_helper.get_min_energy_structure(structures_list))
        
        if display_training_structs:
            for struct in training_dataset_structures_list:
                print('\nUsed struct :')
                struct.print(True)
                
        return training_dataset_structures_list
    
    #Generates a list of structures for testing.
    def generate_structures_for_testing_data(self, structure_name_to_object_map):
        testing_dataset_structures_list = []
        for structure_object in structure_name_to_object_map.values():
            if not structure_object in self.training_dataset_structures_list_:
                testing_dataset_structures_list.append(structure_object)
        return testing_dataset_structures_list
    
    #Generate data vectors
    def generate_data_vectors(self, type_dataset):
        X_list = []
        Y_list = []
        if type_dataset == 'train':
            dataset_structures_list = self.training_dataset_structures_list_
        else:
            dataset_structures_list = self.testing_dataset_structures_list_
        for structure_object in dataset_structures_list:
            correlations = [x.correlation_ for x in structure_object.clusters_list_]
            multiplicities = [x.multiplicity_ for x in structure_object.clusters_list_]
            X_list.append([a*b for a,b in zip(correlations, multiplicities)])
            Y_list.append([structure_object.total_energy_])
        return np.array(X_list), np.array(Y_list)
    
    def get_lasso_object(self, structure_name_to_object_map):
        '''
        We will take out some of the points from testing data and include them
        in training data. Every point in the training data will have certain 
        weight attached to it. To integrate the weights in the loss function,
        we will multiply all the features of a sample with the given weight
        and also the corresponding y value.
        '''
        # Getting 8 random indices from testing dataset.
        indices = np.random.randint(np.shape(self.Y_testing_data_)[0], size=8)
        # Adding the samples to training dataset with weight = 0.65 (It will be
        # squared while calculating loss).
        for index in indices:
            self.X_training_data_ = np.vstack((self.X_training_data_, 
                                               0.65*self.X_testing_data_[index]))
            self.Y_training_data_ = np.vstack((self.Y_training_data_, 
                                               0.65*self.Y_testing_data_[index]))
        # Removing from testing data
        self.X_testing_data_ = np.delete(self.X_testing_data_, indices, 0)
        self.Y_testing_data_ = np.delete(self.Y_testing_data_, indices, 0)
        '''
        Dataset is now ready.
        '''
        print('--------------------------------------------------------------')
        print('Lasso :')
        alphas = [0.0001, 0.0003, 0.0007, 0.001, 0.003, 0.007, 0.01, 0.03, 0.07]
        lasso_list = []
        for alpha_ in alphas:
            lasso = Lasso(alpha=alpha_)
            lasso.fit(self.X_training_data_, self.Y_training_data_)
            train_score=lasso.score(self.X_training_data_, self.Y_training_data_)
            lasso_list.append((train_score, lasso))
        lasso_list.sort(key = lambda x: x[0], reverse = True)
        lasso_best = lasso_list[0][1]
        test_score=lasso_best.score(self.X_testing_data_, self.Y_testing_data_)
        print("\nLasso test score =", test_score)
        print("\nFor training data:\n")
        results = pd.DataFrame({'Actual':self.Y_training_data_[:,0], 'Predicted':lasso_best.predict(self.X_training_data_)[:]})
        print(tabulate(results, headers='keys', tablefmt='psql'))
        print("\nFor testing data:\n")
        results = pd.DataFrame({'Actual':self.Y_testing_data_[:,0], 'Predicted':lasso_best.predict(self.X_testing_data_)[:]})
        print(tabulate(results, headers='keys', tablefmt='psql'))
        print("\nCECs =\n", lasso_best.coef_.reshape(6,1))
        return lasso_best
    
    def get_lr_object(self, structure_name_to_object_map):
        print('--------------------------------------------------------------')
        print('Linear Regression :')
        lr = LinearRegression()
        lr.fit(self.X_training_data_, self.Y_training_data_)
        lr_test_score=lr.score(self.X_testing_data_, self.Y_testing_data_)
        print("\nLR test score =",lr_test_score)
        print("\nFor training data:\n")
        results = pd.DataFrame({'Actual':self.Y_training_data_[:,0], 'Predicted':lr.predict(self.X_training_data_)[:,0]})
        print(tabulate(results, headers='keys', tablefmt='psql'))
        print("\nFor testing data:\n")
        results = pd.DataFrame({'Actual':self.Y_testing_data_[:,0], 'Predicted':lr.predict(self.X_testing_data_)[:,0]})
        print(tabulate(results, headers='keys', tablefmt='psql'))
        print("\nCECs =\n", lr.coef_.reshape(6,1))
        return lr
        
    def get_matinv_object(self, structure_name_to_object_map):
        print('--------------------------------------------------------------')
        print('Matrix Inversion:')
        matrix_inversion_object = matrix_inversion(structure_name_to_object_map,self.X_testing_data_,self.Y_testing_data_)
        print('\nCECs = \n', matrix_inversion_object.CEC_best_)
        return matrix_inversion_object
        
    
        