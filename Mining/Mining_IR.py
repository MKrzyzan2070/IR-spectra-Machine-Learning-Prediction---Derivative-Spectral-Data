# This function processes the input text and separates the data based on the physical state (gas, liquid, or solid),
# as this information is important for analysis.
# Please note that the function takes the state as an input parameter. The possible states are: gas, liquid, solid.

def mining(state):
    import re
    doc = open("Download_IR_NIS/All_the_IR", "r")
    All_IR = doc.readlines()
    doc.close()

    # It's important to consider that different words may describe the same state: e.g., gas and GAS.
    # Notes: 'solution' and 'liquid' are treated as equivalent, as are 'film' and 'liquid', and 'thin' refers to 'liquid'.
    # 'vapor' and 'gas' are also considered the same.
    # 'VISC', 'SEMI', and 'melted' are treated as solid states.
    # The keyword 'NEAT' is discarded.

    # state dict done manually based on list_of state
    state_dict = {"gas": ['gas', 'GAS', 'VAPOR', 'GAS (VAPOR)'],
                  "liquid": ['SOLUTION', 'LIQUID', 'FILM', 'liquid', 'THIN'],
                  "solid": ['SOLID', 'solid', 'VISC', 'SEMI', 'MELTED', 'SSOLID', 'MELT']}

    # Let's make a cool list of indices from and to, so that we know, where to start and begin with out text
    from_to_dict = {"from": [], "to": []}
    from_num = 0
    to_num = 0
    index = -1
    for line in All_IR:
        index += 1
        line = line.strip("\n")
        if line == "NEW NEW NEW" or line == "END END END":
            if line == "NEW NEW NEW":
                from_num = index
            elif line == "END END END":
                to_num = index
            if from_num < to_num:
                from_to_dict["from"].append(from_num)
                from_to_dict["to"].append(to_num)
        else:
            pass

    # The final output dictionary will consist of a list of lists.
    # Each list within a specific state (e.g., gas, liquid, or solid) represents lines corresponding to a given chemical.

    final_dict = {"gas": [], "liquid": [], "solid": []}
    list_to_add = []
    index = -1
    #This number is the index in final_dict = the number of individual species in a row
    number = 0
    found_key_of_dict_before = ""
    found_key_of_dict = ""

    # The underlying logic is that while iterating through the lines, they are added to the list_to_add.
    # By maintaining the index and comparing it to the dictionary (from_to), we can determine the boundaries between species.
    # Additionally, while processing the lines, if a line represents the state of the species, we can use the dictionary to
    # assign it to the proper category. Lines containing "NO IR DISCOVERED" are skipped, so there's no need to worry about those.

    for line in All_IR:
        line = line.strip("\n")
        index += 1
        if index > from_to_dict["from"][number] and index < from_to_dict["to"][number]:
            ##############################
            list_to_add.append(line)
            ##############################

            ###########
            # WOW, SO APPARENTLY NOTHING WORKS IF IT'S \s+\w+, but it's ok, when \w+ is applied !!!!
            ###########
            
            the_find = re.findall("##STATE=(\w+)", line)
            if len(the_find) > 0:
                for key in state_dict:
                    if the_find[0] in state_dict[key]:
                        found_key_of_dict = key
                    else:
                        continue
        elif index >= from_to_dict["to"][number]:
            if len(list_to_add) > 5:

                #########################
                final_dict[found_key_of_dict].append(list_to_add)
                #########################

            number += 1
            list_to_add = []
    
    if state == "gas" or state == "liquid" or state == "solid":
        return final_dict[state]
    elif state == "solid & liquid" or state == "liquid & solid":
        output_list = final_dict["solid"] + final_dict["liquid"]
        return output_list
    elif state == "solid & gas" or state == "gas & solid":
        output_list = final_dict["solid"] + final_dict["gas"]
        return output_list
    elif state == "gas & liquid" or state == "liquid & gas":
        output_list = final_dict["gas"] + final_dict["liquid"]
        return output_list
    elif state == "all":
        output_list = final_dict["gas"] + final_dict["liquid"] + final_dict["solid"]
        return output_list
    else:
        print("Wrong state as input")
        exit()
        return None

# To filter out non-organic compounds or unusual organics with elements like Ti, a function is needed that processes
# the list of lists obtained from the mining() function and outputs a new list of lists containing only organic compounds.
# It's important to specify the state of the molecule in the function, so the appropriate list of lists (or list of lines)
# can be extracted from the dictionary.

def only_organics(state):
    import re
    # There needs to be a list of accepted elements. Added manually
    allowed_elements = ["C", "S", "H", "O", "N", "F", "Br", "Cl", "I", "B", "Si", "Na", "K", "Mg", "Ca"]

    the_list = mining(state)
    new_list_of_molecules = []

    ###############################################################
    index = -1
    list_index_approved_molecules = []
    list_index_NOT_approved_molecules = []
    for element in the_list:
        index += 1
        for line in element:
            the_find = re.findall("##MOLFORM=(\w+)", line)
            if len(the_find) > 0:
                mol_find = re.findall("([A-Za-z]+)\s*\d*", the_find[0])
                not_allowed_check = 0
                carbon_check = False
                for atom in mol_find:
                    if atom not in allowed_elements:
                        not_allowed_check += 1
                    if atom == "C":
                        carbon_check = True
                if not_allowed_check == 0 and carbon_check is True:
                    list_index_approved_molecules.append(index)
                else:
                    list_index_NOT_approved_molecules.append(index)

    index = -1
    for element in the_list:
        index += 1
        if index in list_index_approved_molecules:
            new_list_of_molecules.append(element)

    doc = open("Mining/Only_organics.txt", "w")
    for molecule in new_list_of_molecules:
        for line in molecule:
            doc.write(line)
            doc.write("\n")
    doc.close()

    return new_list_of_molecules

#only_organics("all")