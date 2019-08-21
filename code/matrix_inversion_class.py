import numpy as np
from structure_helper_class import structure_helper

class matrix_inversion:
    
    def __init__(self, structure_name_to_object_map):
        self.X_matrices_list_, self.Y_matrices_list_ = self.generate_matrices(structure_name_to_object_map)
        self.X_inverted_matrices_list_ = self.generate_inverted_matrices()
        self.M_ = self.generate_M()
        
    def generate_matrices(self, structure_name_to_object_map):
        #Getting compositions
        composition_ratio_to_structure_names_list_map = structure_helper.get_composition_ratio_to_structure_names_list_map(structure_name_to_object_map.values())
        #Definite compositions to be included
        definite_compositions = {0.0, 0.33, 0.50, 0.67, 1.00}
        
        composition_ratio_to_min_energy_structure_map = {}
        
        for composition, structures_names_list in composition_ratio_to_structure_names_list_map.items():
            structures_list = []
            for structure_name in structures_names_list:
                structures_list.append(structure_name_to_object_map[structure_name])
            composition_ratio_to_min_energy_structure_map[composition] = structure_helper.get_min_energy_structure(structures_list)
            
        X_matrices_list = []
        Y_matrices_list = []
        
        X_compulsory = []
        Y_compulsory = []
        
        for composition, structure_object in composition_ratio_to_min_energy_structure_map.items():
            if not round(composition,2) in definite_compositions:
                continue
            correlations = [x.correlation_ for x in structure_object.clusters_list_]
            multiplicities = [x.multiplicity_ for x in structure_object.clusters_list_]
            X_compulsory.append([a*b for a,b in zip(correlations, multiplicities)])
#            X_compulsory.append(correlations)
            Y_compulsory.append([structure_object.total_energy_])
            
        for composition, structure_object in composition_ratio_to_min_energy_structure_map.items():
            if round(composition,2) in definite_compositions:
                continue
            correlations = [x.correlation_ for x in structure_object.clusters_list_]
            multiplicities = [x.multiplicity_ for x in structure_object.clusters_list_]
#            X_compulsory.append(correlations)
            X_compulsory.append([a*b for a,b in zip(correlations, multiplicities)])
            Y_compulsory.append([structure_object.total_energy_])
            X_matrices_list.append(np.array(X_compulsory))
            Y_matrices_list.append(np.array(Y_compulsory))
            X_compulsory.pop()
            Y_compulsory.pop()
            
        return X_matrices_list, Y_matrices_list
    
    def generate_inverted_matrices(self):
        X_inverted_matrices_list = []
        for matrix in self.X_matrices_list_:
            try:
                print('\nMatrix to be inverted :')
                print(matrix)
                X_inverted_matrices_list.append(np.linalg.inv(matrix))
            except:
                print('Found a singular matrix', matrix)
        return X_inverted_matrices_list
    
    def generate_M(self):
        print('\nCECs')
        print([np.matmul(X,Y) for X,Y in zip(self.X_inverted_matrices_list_, self.Y_matrices_list_)])
        return [np.matmul(X,Y) for X,Y in zip(self.X_inverted_matrices_list_, self.Y_matrices_list_)]

    def predict(self, X_testing_data):
        print('Matrix inversion results : ')
        result = []
#        for m in self.M_:
        try:
            for matrix in X_testing_data:
                result.append(np.matmul(matrix.reshape(1,6), self.M_[1]))
        except:
            print(np.shape(X_testing_data[0].reshape(1,6)))
            print(np.shape(self.M_[0]))
        return result