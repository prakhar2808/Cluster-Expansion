from structure_helper_class import structure_helper

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

class matrix_inversion:
    
    def __init__(self, structure_name_to_object_map, X_testing_data, Y_testing_data):
        self.X_matrices_list_, self.Y_matrices_list_ = self.generate_matrices(structure_name_to_object_map)
        self.X_inverted_matrices_list_ = self.generate_inverted_matrices()
        self.CEC_list_ = self.generate_CEC_list()
        self.CEC_best_ = self.get_best_CEC(X_testing_data, Y_testing_data, True)
        
    def generate_matrices(self, structure_name_to_object_map):
        #Getting compositions
        composition_ratio_to_structure_names_list_map = structure_helper.get_composition_ratio_to_structure_names_list_map(structure_name_to_object_map.values())
        #Definite compositions to be included
        definite_compositions = {0.0, 0.33, 0.50, 0.67, 1.00}
        
        composition_ratio_to_min_energy_structure_map = {}
        #Generating map from composition ratio to min. energy structure
        for composition, structures_names_list in composition_ratio_to_structure_names_list_map.items():
            structures_list = []
            for structure_name in structures_names_list:
                structures_list.append(structure_name_to_object_map[structure_name])
            composition_ratio_to_min_energy_structure_map[composition] = structure_helper.get_min_energy_structure(structures_list)
        
        #List of matrices formed by selecting the minimum energy structure
        #from 6 of the compositions as each matrix is of size 6 x 6.
        X_matrices_list = []
        Y_matrices_list = []
        #The structures which are to be always included.
        X_compulsory = []
        Y_compulsory = []
        #Generating matrix for the structures which are always to be included.
        for composition, structure_object in composition_ratio_to_min_energy_structure_map.items():
            if not round(composition,2) in definite_compositions:
                continue
            correlations = [x.correlation_ for x in structure_object.clusters_list_]
            multiplicities = [x.multiplicity_ for x in structure_object.clusters_list_]
            X_compulsory.append([a*b for a,b in zip(correlations, multiplicities)])
            Y_compulsory.append([structure_object.total_energy_])
        #Adding a structure from remaining compositions one at a time and 
        #appending the matrix to the X_matrices_list and Y_matrices_list.
        for composition, structure_object in composition_ratio_to_min_energy_structure_map.items():
            if round(composition,2) in definite_compositions:
                continue
            correlations = [x.correlation_ for x in structure_object.clusters_list_]
            multiplicities = [x.multiplicity_ for x in structure_object.clusters_list_]
            X_compulsory.append([a*b for a,b in zip(correlations, multiplicities)])
            Y_compulsory.append([structure_object.total_energy_])
            X_matrices_list.append(np.array(X_compulsory))
            Y_matrices_list.append(np.array(Y_compulsory))
            X_compulsory.pop()
            Y_compulsory.pop()
            
        return X_matrices_list, Y_matrices_list
    
    def generate_inverted_matrices(self, display_matrices = False):
        X_inverted_matrices_list = []
        for matrix in self.X_matrices_list_:
            try:
                if display_matrices:
                    print('\nMatrix to be inverted :')
                    print(matrix)
                inv_matrix = np.linalg.inv(matrix)
                if display_matrices:
                    print('\nInverted matrix :')
                    print(inv_matrix)
                X_inverted_matrices_list.append(inv_matrix)
            except:
                print('\n Warning : Failed inverting. Skipping.\n', matrix)
        return X_inverted_matrices_list
    
    def generate_CEC_list(self, display_CECs = False):
        CEC_list = [np.matmul(X,Y) for X,Y in zip(self.X_inverted_matrices_list_, self.Y_matrices_list_)]
        if display_CECs:
            print('\nCECs:\n')
            for CEC in CEC_list:
                print(CEC)
        return CEC_list

    def get_best_CEC(self, X_testing_data, Y_testing_data, print_results = False):
        test_score_to_CEC_index_map_ = {}
        predictions_list = []
        for index in range(len(self.CEC_list_)):
            predictions = []
            for row in X_testing_data:
                predictions.append(np.matmul(row.reshape(1,6), self.CEC_list_[index]))
            predictions_list.append(predictions)
            score = r2_score(Y_testing_data, np.array(predictions).reshape(np.shape(Y_testing_data)), sample_weight=None, multioutput='variance_weighted')
            test_score_to_CEC_index_map_[score] = index
        max_score = max(test_score_to_CEC_index_map_.keys())
        if print_results:
            predictions = (np.array(predictions_list[test_score_to_CEC_index_map_[max_score]])).reshape(np.shape(Y_testing_data))
            print('\nMatrix Inversion test score = ', max_score)
            results = pd.DataFrame({'Actual':Y_testing_data[:,0], 'Predicted':predictions[:,0]})
            print('\nFor testing data:\n')
            print(results)
        return self.CEC_list_[test_score_to_CEC_index_map_[max_score]]