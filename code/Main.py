from parser_class import Parser
from structure_class import structure

def main():
    lattice_type = str(input("Enter lattice type : "))
    file_name = str(input("Enter file name as <filename.txt> : "))
    max_distance = float(input("Enter max. points distance : "))
    
    #Parsing the above entered file to get the list of parameters for all structures.
    structures_parameters_list = Parser.parse(lattice_type, file_name)
    
    structures_list = []
    #Getting the list of all the structure objects.
    for parameters in structures_parameters_list:
        structures_list.append(structure(parameters, str(max_distance)))
    #Printing the structures
    for structure_object in structures_list:
        structure_object.print()
    

# Calling main function
if __name__ == "__main__":
    main()