from structure_helper_class import structure_helper

class model_train:
    
    #Constructor.
    def __init__(self, structure_name_to_object_map):
        self.training_dataset_structures_list = self.generate_structures_for_training_data(structure_name_to_object_map)
        self.testing_dataset_structures_list = self.generate_structures_for_testing_data(structure_name_to_object_map)
    
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
            if not structure_object in self.training_dataset_structures_list:
                testing_dataset_structures_list.append(structure_object)
        return testing_dataset_structures_list
                