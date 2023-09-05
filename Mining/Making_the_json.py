# This function is designed to find the largest delta X value within a user-specified upper limit,
# ensuring that the resulting dataframe has a consistent number of rows and columns across different molecules.
# Although this approach may result in converting high-resolution spectra to lower resolution to match with other spectra,
# it allows users to set the desired upper limit for the resolution.

# It's important to note that the code in this function is derived from the Mining_IR.py script, which is time-consuming.
# Therefore, it was decided to store the final dictionary as a JSON file. The dictionary contains the following keys:
# name_list, cas_list, yunits_list, xunits_list, xfactor_list, yfactor_list, deltaX_list, IR_signals, IR_signals_value

# The IR_signals and IR_signals_value are not finished products. They contain a list of signals that a molecule has
# (similar to the format in a txt file). For a given molecule, the IR_signals may look like [3200, 3198, 3196...],
# and the corresponding IR_signals_value may look like [[0.987, 0.769...], [...], [...] ....].

# The deltax value is also stored, which will later be used to create a uniform delta across all molecules.
# This uniformity is essential for creating a dataframe from the data.
# Instead of looking for the max delta X, the delta X of each molecule is checked to ensure it is within the
# user-specified resolution_upper_limit.

def making_json(state, lower_bound, upper_bound, resolution_upper_limit):
    import re
    from Mining import Mining_IR
    print("Making only organics file from the All_IR file...")
    #Ok, it's about time to make a dataframe
    molecule_list = Mining_IR.only_organics(state)
    print("Completed only organics file complete")
    print("Creating the dictionary.json file...")
    ###########################################################################
    ###########################################################################
    name_list = []
    cas_list = []
    yunits_list = []
    xunits_list = []
    xfactor_list = []
    yfactor_list = []
    deltaX_list = []
    IR_signals = []
    IR_signals_value = []
    total_list = [name_list, cas_list, yunits_list, xunits_list, xfactor_list, yfactor_list, IR_signals, IR_signals_value]
    resolution_upper_limit = float(resolution_upper_limit)
    ###########################################################################
    ###########################################################################

    # This part focuses on obtaining the desired index list. If the delta X is greater than the resolution limit,
    # it will not be included in the index_list.
    # Additionally, it is worth noting that not all molecules have a Delta X value for some reason.
    # Therefore, the index_list is crucial, as it will be used later to filter out unwanted or poorly characterized molecules.

    for molecule in molecule_list:
        mini_check = 0
        IR_signals_mol = []
        IR_signals_value_mol = []
        #this variable checks againt doubles
        the_end = 0
        dont_add_this_molecule = False
        for line in molecule:
            the_find = re.findall("##NAMES=(.+)", line)
            if len(the_find) > 0 and the_end == 0:
                name_list.append(the_find[0])
                mini_check = 1
            the_find = re.findall("##CAS REGISTRY NO=(.+)", line)
            if len(the_find) > 0 and the_end == 0:
                cas_list.append(the_find[0])
            the_find = re.findall("##YUNITS=(.+)", line)
            if len(the_find) > 0 and the_end == 0:
                yunits_list.append(the_find[0])
            the_find = re.findall("##XUNITS=(.+)", line)
            if len(the_find) > 0 and the_end == 0:
                xunits_list.append(the_find[0])
            the_find = re.findall("##XFACTOR=(.+)", line)
            if len(the_find) > 0 and the_end == 0:
                xfactor_list.append(the_find[0])
            the_find = re.findall("##YFACTOR=(.+)", line)
            if len(the_find) > 0 and the_end == 0:
                yfactor_list.append(the_find[0])
            the_find = re.findall("^\d.*", line)
            if len(the_find) > 0 and the_end == 0:
                check_find = re.findall("[a-zA-Z]+", line)
                if len(check_find) == 0:
                    numbers = the_find[0].split()
                    if len(numbers) > 1:

                        ##########
                        # Yeah, so sometimes there's this: - between numbers at random. Let's just skip those:
                        ##########
                        try:
                            primary_number = round(float(numbers[0]),3)
                        except:
                            dont_add_this_molecule = True
                            continue
                        if primary_number > 50:
                            if primary_number > lower_bound and primary_number < upper_bound:
                                try:
                                    numbers = [round(float(x),3) for x in numbers]
                                except:
                                    dont_add_this_molecule = True
                                    continue
                                IR_signals_mol.append(primary_number)
                                IR_signals_value_mol.append(numbers[1:])

                        #Here is the micrometers instance. It needs to be converted:
                        else:
                            primary_number = round(10000/primary_number,3)
                            if primary_number > lower_bound and primary_number < upper_bound:
                                try:
                                    numbers = [round(float(x),3) for x in numbers]
                                except:
                                    dont_add_this_molecule = True
                                    continue
                                IR_signals_mol.append(primary_number)
                                IR_signals_value_mol.append(numbers[1:])

                    #Of course it couldn't be just space
                    numbers = the_find[0].split("+")
                    if len(numbers) > 1:
                        try:
                            primary_number = round(float(numbers[0]),3)
                        except:
                            dont_add_this_molecule = True
                            continue
                        if primary_number > 50:
                            if primary_number > lower_bound and primary_number < upper_bound:
                                try:
                                    numbers = [round(float(x),3) for x in numbers]
                                except:
                                    dont_add_this_molecule = True
                                    continue
                                IR_signals_mol.append(primary_number)
                                IR_signals_value_mol.append(numbers[1:])
                        else:
                            primary_number = round(10000/primary_number,3)
                            if primary_number > lower_bound and primary_number < upper_bound:
                                try:
                                    numbers = [round(float(x),3) for x in numbers]
                                except:
                                    dont_add_this_molecule = True
                                    continue
                                IR_signals_mol.append(primary_number)
                                IR_signals_value_mol.append(numbers[1:])

            # This exists, because sometimes there's no NAME, but there's always a TITLE
            if mini_check == 0 and line == "##END=":
                for molecule_line in molecule:
                    the_find = re.findall("##TITLE=(.+)", molecule_line)
                    if len(the_find) > 0:
                        name_list.append(the_find[0])
                        mini_check = 1
                        the_end += 1

            if line == "##END=" and dont_add_this_molecule is False:
                IR_signals.append(IR_signals_mol)
                IR_signals_value.append(IR_signals_value_mol)

            #I think it's its fault it really might be, holly molly:
            if line == "##END=" and dont_add_this_molecule is False:
                len_list = [len(list) for list in total_list]
                smaller = min(len_list)
                for list in total_list:
                    if len(list) > smaller:
                        list.pop(-1)

    # Making floats:
    xfactor_list = [float(x) for x in xfactor_list]
    yfactor_list = [float(x) for x in yfactor_list]

    repair_list_0 = [name_list, cas_list, yunits_list, xunits_list, xfactor_list, yfactor_list, IR_signals, IR_signals_value]

    index = -1
    index_list = []
    for signal_list in IR_signals:
        index += 1
        if len(signal_list) > 0:
            index_list.append(index)

    repair_list_1 = []
    for lis in repair_list_0:
        new_list = [lis[i] for i in index_list]
        repair_list_1.append(new_list)
    repair_list_0 = []

    new_IR_signal_list = []
    new_IR_signalValue_list = []
    index = -1
    for IR_signal_list_for_element in repair_list_1[-2]:
        index += 1
        IR_signalValue_list_for_element = repair_list_1[-1][index]
        #The thing above is the list of lists, while IR_signal is a list
        first = IR_signal_list_for_element[0]
        last = IR_signal_list_for_element[-1]
        if first > last:
            new_IR_signal_list_for_element = IR_signal_list_for_element[::-1]
            new_IR_signal_list.append(new_IR_signal_list_for_element)

            new_IR_signalValue_list_for_element = []
            #Now the regular signal list:
            for signal_value_sub_list in IR_signalValue_list_for_element:
                new_signal_value_sub_list = signal_value_sub_list[::-1]
                new_IR_signalValue_list_for_element.append(new_signal_value_sub_list)
            #and reverse again, this time the whole:
            new_IR_signalValue_list_for_element = new_IR_signalValue_list_for_element[::-1]
            new_IR_signalValue_list.append(new_IR_signalValue_list_for_element)

        else:
            new_IR_signal_list.append(IR_signal_list_for_element)
            new_IR_signalValue_list.append(IR_signalValue_list_for_element)

    #[name_list, cas_list, yunits_list, xunits_list, xfactor_list, yfactor_list, IR_signals, IR_signals_value]

    #IR signals:
    repair_list_1[-2] = new_IR_signal_list

    #IR signals value:
    repair_list_1[-1] = new_IR_signalValue_list

    index = -1
    index_list = []
    for element in repair_list_1[-2]:
        index += 1
        index_list.append(index)

        how_many_1 = float(len(repair_list_1[-1][index][0]))
        how_many_2 = float(len(repair_list_1[-1][index][-1]))

        # Element is a list so I have no idea, why I get these errors. Sometimes they just disappear so yeah
        delta_1 = element[1] - element[0]
        delta_2 = element[-1] - element[-2]
        div1 = abs(delta_1/how_many_1)
        div2 = abs(delta_2/how_many_2)
        
        #Here the resolution_upper_limit comes into play, as it is checked if the delta is lower than this limit
        
        if abs(div1) < resolution_upper_limit and abs(div2) < resolution_upper_limit:
            ###################### MAKE COMMENT HERE
            sub_deltaX_list = []

            # Again we have error, but element is a list so it works ok
            # Sometimes this error pops out: Expected type 'Size' got 'float' instead
            length = len(element)
            num1 = -1
            num2 = 0
            # Warning this could be bad !!!
            for i in range(length):
                if num2 < length - 1:
                    num1 += 1
                    num2 += 1
                    delta = float(element[num2]) - float(element[num1])
                    how_many = float(len(repair_list_1[-1][index][num1]))
                    sub_deltaX_list.append(delta/how_many)
                else:
                    delta = float(element[num2]) - float(element[num1])
                    how_many = float(len(repair_list_1[-1][index][num1]))
                    sub_deltaX_list.append(delta/how_many)
            deltaX_list.append(sub_deltaX_list)
        else:
            index_list.remove(index)

    repair_list_2 = []
    for lis in repair_list_1:
        new_list = [lis[i] for i in index_list]
        repair_list_2.append(new_list)

    repair_list_1 = []
    repair_list_2.append(deltaX_list)

    ##################################################
    #Just in case it's better to make it slightly bigger, so that there won't random problems later
    deltax = 1.05*resolution_upper_limit
    ##################################################

    # One last consideration is related to the range of values in the spectra. For example, if the lower limit is 500 and a spectrum starts at 700,
    # this could cause issues later when comparing spectra with different starting values.
    # To prevent this, it is important to ensure that the first value of the IR spectrum (in cm-1) is in close proximity to the lower bound.

    index = -1
    index_list = []
    IR_signal = repair_list_2[6]
    for signal_list in IR_signal:
        index += 1
        first = signal_list[0]
        last = signal_list[-1]
        if lower_bound + 9*deltax > first and upper_bound - 9*deltax < last:
            index_list.append(index)
            
    repair_list_3 = []
    for lis in repair_list_2:
        new_list = [lis[i] for i in index_list]
        repair_list_3.append(new_list)
    repair_list_2 = []

    the_final_dict = {"Name": repair_list_3[0], "CAS": repair_list_3[1], "Y_factor": repair_list_3[5], "Y_unit": repair_list_3[2],
                      "IR_signal": repair_list_3[6], "IR_signal_value": repair_list_3[7], "Delta_X": repair_list_3[8],
                      "Uniform_Delta_X": deltax, "lower_bound": lower_bound, "upper_bound": upper_bound}

    #Aaaaaand the final action:
    import json
    json_object = json.dumps(the_final_dict, indent=len(the_final_dict))
    with open("Mining/Dictionary.json", "w") as doc:
        doc.write(json_object)

    print("The json dictionary is complete")

    return None

######################################################################
#making_json("solid & liquid", 600, 3100, 5.05)