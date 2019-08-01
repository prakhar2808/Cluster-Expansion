class structure_helper:
    
    def get_all_elements_from_structures_list(structures_list):
        all_elements = set()
        for structure_object in structures_list:
            for element in structure_object.source_positions_.keys():
                all_elements.add(element)
        return list(all_elements)
    
    def get_composition_ratio(structure_object, all_elements):
        x = 0.0
        y = 0.0
        if all_elements[0] in structure_object.source_positions_:
            x = len(structure_object.source_positions_[all_elements[0]])
        if all_elements[1] in structure_object.source_positions_:
            y = len(structure_object.source_positions_[all_elements[1]])
        return x/(x+y)
    
    def get_composition_ratio_to_structure_names_list_map(structures_list):
        #Getting the union of elements present in all structures
        all_elements = structure_helper.get_all_elements_from_structures_list(structures_list)
    
        #Getting composition ratios for all structures:
        composition_ratio_to_structure_names_list_map = {}
        for structure_object in structures_list:
            composition_ratio = structure_helper.get_composition_ratio(structure_object, all_elements)
            if not composition_ratio in composition_ratio_to_structure_names_list_map:
                composition_ratio_to_structure_names_list_map[composition_ratio] = []
            composition_ratio_to_structure_names_list_map[composition_ratio].append(structure_object.name_)
        return composition_ratio_to_structure_names_list_map
    
    def get_min_energy_structure(structures_list):
        structures_list.sort(key=lambda x: x.total_energy_)
        return structures_list[0]
        
        