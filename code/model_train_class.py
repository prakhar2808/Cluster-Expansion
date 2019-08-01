from structure_helper_class import structure_helper

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression

class model_train:
    
    #Constructor.
    def __init__(self, structure_name_to_object_map):
        self.training_dataset_structures_list_ = self.generate_structures_for_training_data(structure_name_to_object_map)
        self.testing_dataset_structures_list_ = self.generate_structures_for_testing_data(structure_name_to_object_map)
        self.X_training_data_, self.Y_training_data_ = self.generate_training_data_vectors('train')
        self.X_testing_data_, self.Y_testing_data_ = self.generate_training_data_vectors('test')
        self.thetas_ = self.get_thetas()
    
    #Generates a list of structures for training.
    def generate_structures_for_training_data(self, structure_name_to_object_map):
        #Getting compositions
        composition_ratio_to_structure_names_list_map = structure_helper.get_composition_ratio_to_structure_names_list_map(structure_name_to_object_map.values())
        #Generating structures for training data
        training_dataset_structures_list = []
        for composition,structures_names_list in composition_ratio_to_structure_names_list_map.items():
            structures_list = []
            for structure_name in structures_names_list:
                structures_list.append(structure_name_to_object_map[structure_name])
            training_dataset_structures_list.append(structure_helper.get_min_energy_structure(structures_list))
        return training_dataset_structures_list
    
    #Generates a list of structures for testing.
    def generate_structures_for_testing_data(self, structure_name_to_object_map):
        testing_dataset_structures_list = []
        for structure_object in structure_name_to_object_map.values():
            if not structure_object in self.training_dataset_structures_list_:
                testing_dataset_structures_list.append(structure_object)
        return testing_dataset_structures_list
    
    #Generate training data vectors
    def generate_training_data_vectors(self, type_dataset):
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
    
    def get_thetas(self):
        lasso = Lasso()
        lasso.fit(self.X_training_data_, self.Y_training_data_)
        train_score=lasso.score(self.X_training_data_, self.Y_training_data_)
        test_score=lasso.score(self.X_testing_data_, self.Y_testing_data_)
        print("Lasso train score =",train_score)
        print("Lasso test score =",test_score)
        print("Lasso coeff =",lasso.coef_)
#        print("For training data:\nActual\tPredicted")
#        print(np.c_[self.Y_training_data_, lasso.predict(self.X_training_data_)])
#        print("For testing data:\nActual\tPredicted")
#        print(np.c_[self.Y_testing_data_, lasso.predict(self.X_testing_data_)])
        
        lr = LinearRegression()
        lr.fit(self.X_training_data_, self.Y_training_data_)
        lr_train_score=lr.score(self.X_training_data_, self.Y_training_data_)
        lr_test_score=lr.score(self.X_testing_data_, self.Y_testing_data_)
        print("LR train score =",lr_train_score)
        print("LR test score =",lr_test_score)
        print("LR coeff =", lr.coef_)
#        print("For training data:\nActual\tPredicted")
#        print(np.c_[self.Y_training_data_, lr.predict(self.X_training_data_)])
#        print("For testing data:\nActual\tPredicted")
#        print(np.c_[self.Y_testing_data_, lr.predict(self.X_testing_data_)])
        return lasso.coef_
        