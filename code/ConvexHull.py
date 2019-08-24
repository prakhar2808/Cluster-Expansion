from structure_helper_class import structure_helper

import matplotlib.pyplot as plt

class convex_hull:
    
    def draw(structure_name_to_object_map):
        #Getting a map from composition ratio to list of structure names
        composition_ratio_to_structure_names_list_map = structure_helper.get_composition_ratio_to_structure_names_list_map(structure_name_to_object_map.values())
        #Generating plotting data
        points_x = []
        points_y = []
        hull_x = []
        hull_y = []
        for composition in sorted(composition_ratio_to_structure_names_list_map.keys()):
            min_energy = 100
            structure_names = composition_ratio_to_structure_names_list_map[composition]
            for name in structure_names:
                points_x.append(composition)
                points_y.append(structure_name_to_object_map[name].total_energy_)
                min_energy = min(min_energy, structure_name_to_object_map[name].total_energy_)
            hull_x.append(composition)
            hull_y.append(min_energy)
        
        plt.scatter(points_x, points_y, marker='.')
        for i in range(len(hull_x)-1):
            plt.plot([hull_x[i], hull_x[i+1]],[hull_y[i], hull_y[i+1]],'k-')
        plt.show()