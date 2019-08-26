from parser_class import Parser
from structure_class import structure
from model_train_class import model_train
from model_train_helper_class import model_train_helper
from ConvexHull import convex_hull


def main():
#    lattice_type = str(input("Enter lattice type : "))
#    file_name = str(input("Enter file name as <filename.txt> : "))
#    max_distance = float(input("Enter max. points distance : "))
    lattice_type = 'bcc'
    elements = ['Ti', 'V']
    max_distance = 3.4
#    elements = ['Sc', 'Ti']
#    max_distance = 3.879794
#    elements = ['Cr', 'Ti']
#    max_distance = 3.394820
#    elements = ['Sc', 'V']
#    max_distance = 3.879794
    file_name = elements[0]+'_'+elements[1]+'.txt'
    
    
    #Parsing the above entered file to get the list of parameters for all structures.
    structures_parameters_list = Parser.parse(lattice_type, file_name)
        
    #Storing map from structure name to structure object for easier access.
    structure_name_to_object_map = {}
    #Getting the list of all the structure objects.
    for parameters in structures_parameters_list:
        try:
            structure_object = structure(parameters, str(max_distance), elements)
            structure_name_to_object_map[structure_object.name_] = structure_object
        except:
            continue                
        
    #Drawing convex hull for all the structures.
#    convex_hull.draw(structure_name_to_object_map)
    
    #Printing the structures
#    for structure_object in structure_name_to_object_map.values():
#        structure_object.print()
        
    model_train_object = model_train(structure_name_to_object_map) 
    
    model_train_helper.verify_predictions(structure_name_to_object_map, 
                                          model_train_object.lasso_object_, 'Lasso')
#    
    model_train_helper.verify_predictions(structure_name_to_object_map, 
                                          model_train_object.lr_object_, 'LR')
    
    model_train_helper.verify_predictions(structure_name_to_object_map, 
                                          model_train_object.matinv_object_, 'Matrix Inversion')
    
# Calling main function
if __name__ == "__main__":
    main()