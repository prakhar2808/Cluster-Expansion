from structure_helper_class import structure_helper

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

class matrix_inversion_alternate:
    
    def __init__(self, structure_name_to_object_map, X_testing_data, Y_testing_data):
        self.X_matrix_, self.Y_matrix_ = self.generate_matrices(structure_name_to_object_map)
        self.X_inverted_matrix_ = self.generate_inverted_matrices()
        self.CEC_ = self.generate_CEC()
        self.predict(X_testing_data, Y_testing_data, True)
        
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
        #The structures which are to be always included.
        X_compulsory = []
        Y_compulsory = []
        #Generating matrix for the structures which are always to be included.
        for composition, structure_object in composition_ratio_to_min_energy_structure_map.items():
            if not round(composition,2) in definite_compositions:
                continue
            correlations = [x.correlation_ for x in structure_object.clusters_list_]
            multiplicities = [x.multiplicity_ for x in structure_object.clusters_list_]
            X_compulsory.append([a*b for a,b in zip(correlations, multiplicities)][1:])
            Y_compulsory.append([structure_object.total_energy_])
                        
        return np.array(X_compulsory), np.array(Y_compulsory)
    
    def generate_inverted_matrices(self, display_matrices = False):
        try:
            if display_matrices:
                print('\nMatrix to be inverted :')
                print(self.X_matrix_)
            inv_matrix = np.linalg.inv(self.X_matrix_)
            if display_matrices:
                print('\nInverted matrix :')
                print(inv_matrix)
            return inv_matrix
        except:
            print('\n Warning : Failed inverting. Skipping.\n', self.X_matrix_)
            return None
    
    def generate_CEC(self, display_CEC = False):
        CEC = np.matmul(self.X_inverted_matrix_, self.Y_matrix_)
        if display_CEC:
            print('\nCECs:\n')
            print(CEC)
        return CEC

    def predict(self, X_testing_data, Y_testing_data, print_results = False):
        predictions = []
        print(np.shape(self.CEC_))
        print(np.shape(X_testing_data[0]))
        for row in X_testing_data:
            predictions.append(np.matmul(row[1:].reshape(1,5), self.CEC_))
        score = r2_score(Y_testing_data, np.array(predictions).reshape(np.shape(Y_testing_data)), sample_weight=None, multioutput='variance_weighted')
        print(score)
        if print_results:
            predictions = (np.array(predictions)).reshape(np.shape(Y_testing_data))
            print('\nMatrix Inversion test score = ', score)
            results = pd.DataFrame({'Actual':Y_testing_data[:,0], 'Predicted':predictions[:,0]})
            print('\nFor testing data:\n')
            print(results)
        