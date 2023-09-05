import pandas as pd


# noinspection PyBroadException
def making_json_2_and_df():
    import json
    import pandas as pd
    # Opening JSON file
    print("Making additional operations on Dictionary json file...")
    with open('Mining/Dictionary.json', 'r') as openfile:
        # Reading from json file
        the_dictionary = json.load(openfile)

    IR_signals = the_dictionary["IR_signal"]
    IR_signal_values = the_dictionary["IR_signal_value"]
    DeltaX_list = the_dictionary["Delta_X"]
    NEW_IR_signals = []
    NEW_IR_signal_values = []

    #Here it's high time to convert IR_signals so that they are a full-blown list
    index = -1
    for molecule in IR_signals:
        index += 1
        sub_index = -1
        sub_NEW_IR_signals = []
        sub_NEW_IR_signals_values = []
        for signal in molecule:
            sub_index += 1
            delta = DeltaX_list[index][sub_index]
            length = len(IR_signal_values[index][sub_index])
            for n in range(length):
                new_signal = signal + delta*n
                value = IR_signal_values[index][sub_index][n]
                sub_NEW_IR_signals.append(new_signal)
                sub_NEW_IR_signals_values.append(value)
        NEW_IR_signals.append(sub_NEW_IR_signals)
        NEW_IR_signal_values.append(sub_NEW_IR_signals_values)

    #Using the factors to uniform all signals:
    Y_factor = the_dictionary["Y_factor"]
    Y_unit = the_dictionary["Y_unit"]
    index = -1
    index_list = []
    for molecule in NEW_IR_signal_values:
        index += 1
        factor = Y_factor[index]
        unit = Y_unit[index]
        check_abs = False

        ##############For now I don't know what to do with reflectance################
        if unit == 'TRANSMITTANCE' or unit == "ABSORBANCE":
            index_list.append(index)
        if unit == "ABSORBANCE":
            check_abs = True

        new_list = []
        if check_abs is False:
            for value in molecule:
                new_value = value*factor
                new_list.append(new_value)
        elif check_abs is True:
            for value in molecule:
                new_value = value*factor
                #################################
                #################################
                new_list.append(10**(-new_value))
                #################################
                #################################

        NEW_IR_signal_values.pop(index)
        NEW_IR_signal_values.insert(index, new_list)

    the_dictionary["IR_signal"] = NEW_IR_signals
    the_dictionary["IR_signal_value"] = NEW_IR_signal_values

    for key in the_dictionary:
        # noinspection PyBroadException
        try:
            the_list = the_dictionary[key]
            new_list = [the_list[i] for i in index_list]
            the_dictionary[key] = new_list
        except:
            continue

    #We don't need that anymore:
    the_dictionary.pop("Delta_X")
    the_dictionary.pop("Y_factor")
    the_dictionary.pop("Y_unit")

    #I'll make both csv file and a mini json file. csv file will store Name, CAS and signals, while
    #json will just store the three numbers: Uniform_Delta_X, and lower/upper bound
    csv_list_key = []
    json_list_key = []
    for key in the_dictionary:
        #Because len() doesn't work on lower and upper bound, because they are numbers not list or string.
        try:
            len(the_dictionary[key])
            csv_list_key.append(key)
        except:
            json_list_key.append(key)

    dict_to_maxi_json = {}
    dict_to_mini_json = {}
    for key in csv_list_key:
        dict_to_maxi_json[key] = the_dictionary[key]
    for key in json_list_key:
        dict_to_mini_json[key] = the_dictionary[key]

    #Reminder of keys in the_dictionary: Name, CAS, IR_signal, IR_signal_value, Uniform_Delta_X, lower_bound, upper_bound
    #Now one last thing is that we should add the SMILES to the dataframe
    df_to_maxi_json = pd.DataFrame(dict_to_maxi_json)
    #And let's make the mini json
    json_object = json.dumps(dict_to_mini_json, indent=len(dict_to_mini_json))
    with open("Mining/Mini_dictionary.json", "w") as doc:
        doc.write(json_object)

    print("Additional operations complete")
    return df_to_maxi_json


# noinspection PyBroadException
def lets_put_a_smile_on_your_face():
    import os
    import numpy as np
    # Yeah, time add SMILES into the csv file
    the_df = making_json_2_and_df()
    import cirpy
    #Just in case if it couldn't get the smiles:
    print("Adding the SMILE string...")
    smiles_list = []
    index = -1
    index_list = []
    length = len(the_df)
    anti_loop = 0

    #############################################################
    #############################################################
    #This whole thing unfortunately takes a million years,
    #so I will do this once and later on, it will just read
    #from the json:
    #############################################################
    #############################################################
    SMILES_from_CSV = True

    if SMILES_from_CSV is False:
        for cas in the_df["CAS"]:
            index += 1
            print(index)
            if index/length > 0.5 and anti_loop == 0:
                print("Half of the SMILE is done")
                anti_loop  = 1
            try:
                smiles = cirpy.resolve(cas, 'smiles')
                if smiles is not None:
                    smiles_list.append(smiles)
                    index_list.append(index)
            except:
                continue

        #So now let's make a pandas Series of SMILES and store it as .csv
        names_list = list(the_df.iloc[index_list]["Name"])
        SMILES_series = pd.Series(data=smiles_list, index=names_list, name="SMILES")
        try:
            os.remove("Mining/SMILES_CSV.csv")
        except:
            pass
        SMILES_series.to_csv("Mining/SMILES_CSV.csv")

    else:
        #Ok so now comes the reading of csv:
        SMILES_series = pd.read_csv("Mining/SMILES_CSV.csv")
        SMILES_series.set_index("Unnamed: 0", inplace=True)
        SMILES_list = []
        for name in the_df["Name"]:
            try:
                smiles_string = str(SMILES_series.loc[name]["SMILES"])
            except:
                smiles_string = np.nan
            SMILES_list.append(smiles_string)

        the_df["SMILES"] = SMILES_list
        #Aaaaand finally making the csv file - NO NO NO, ok so csv doesn't work that well, so let's make json
        the_df.dropna(inplace=True)
        the_df.reset_index(drop=True, inplace=True)
        final_dict = the_df.to_dict()
        #print(the_df)

        import json
        json_object = json.dumps(final_dict, indent=len(final_dict))
        with open("Mining/Maxi_dictionary.json", "w") as doc:
            doc.write(json_object)


        print("Assigning SMILE string complete")

    os.remove("Mining/Dictionary.json")
    return None

######################################################
#lets_put_a_smile_on_your_face()