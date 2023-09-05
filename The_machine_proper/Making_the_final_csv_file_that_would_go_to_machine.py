import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os
import json
from progress.bar import Bar
pd.set_option("display.max_columns", None)

def Creating_Uniform_Plain_CSV():

    with open("Mining/Maxi_dictionary.json", 'r') as openfile:
        the_dict = json.load(openfile)
    with open("Mining/Mini_dictionary.json", 'r') as openfile:
        mini_dict = json.load(openfile)

    the_df = pd.DataFrame(the_dict)
    the_df.reset_index(inplace=True, drop=True)

    #The user determined upper and lower bounds are not the same as what will come out of it in the end:
    lower_bound_list = np.array([])
    upper_bound_list = np.array([])
    for index in the_df.index:
        signal_list = np.array(the_df["IR_signal"].loc[index])
        lower_bound_list = np.append(lower_bound_list, signal_list[0])
        upper_bound_list = np.append(upper_bound_list, signal_list[-1])
    true_lower_bound = np.max(lower_bound_list)
    true_upper_bound = np.min(upper_bound_list)

    # Creating the reference signal list, with the help of previously derived lower and upper bound and the user-stated deltaX
    ref_sig_list = np.array([])
    deltax = mini_dict["Uniform_Delta_X"]
    signal = true_lower_bound
    upper_limit = true_upper_bound
    while signal < upper_limit:
        ref_sig_list = np.append(ref_sig_list, round(signal, 2))
        signal += deltax

    # IR_signal and IR_signal_value:
    value_list_of_ref_sig = []
    
    # For each compound - index in the df:
    progress_bar = Bar('Creating Uniform signals: ', max=len(the_df.index))
    progress_bar.start()
    for index_df in the_df.index:
        compound_value_list_of_ref_sig = np.array([])
        signal_list = np.array(the_df["IR_signal"].loc[index_df])
        for index in range(len(ref_sig_list) - 1):
            ref_signal = ref_sig_list[index]
            ref_signal_2 = ref_sig_list[index + 1]

            # For each incremental boundary (ref_signal and ref_signal_2), a snippet from the signal_list is determined.
            # This snippet is stored as a list of indices. When the snippet has a length greater than one,
            # the mean value of the corresponding signal_values is calculated.

            count = 0
            signal_index = -1
            signal_index_list = np.array([])
            for signal in signal_list:
                signal_index += 1
                if signal > ref_signal and signal <= ref_signal_2:
                    signal_index_list = np.append(signal_index_list, signal_index)
                    count += 1

            if count == 1:
                compound_value_list_of_ref_sig = np.append(compound_value_list_of_ref_sig, np.array(the_df["IR_signal_value"].loc[int(index_df)][int(signal_index_list[0])]))
            elif count > 1:
                # Getting the values of signal_values using the signal_index list
                value_list = np.array([])
                for signal_index in signal_index_list:
                    value = the_df["IR_signal_value"].loc[int(index_df)][int(signal_index)]
                    value_list = np.append(value_list, value)
                    
                compound_value_list_of_ref_sig = np.append(compound_value_list_of_ref_sig, np.mean(value_list))

        progress_bar.next()
        # The resulting data structure is a list of lists, where each list corresponds to a column representing the signal value.
        # This can be more easily visualized using a graph or another visual representation.

        value_list_of_ref_sig.append(compound_value_list_of_ref_sig)

    progress_bar.finish()

    # IMPORTANT NOTE: The algorithm above implies that a value like 800 represents the range from 800 to 800 + delta X.
    # In cases where multiple matches occur, this range corresponds to the mean of those values.
    # Consequently, we end up with one less reference signal than previously.
    # To reduce confusion, it is recommended to use new and clearer column names that represent the ranges,
    # such as "800 to 800 + delta X."

    the_super_df = the_df[["Name", "CAS", "SMILES"]]
    the_super_dictionary = {}
    
    # -1 is simply the consequence of things explained above:
    column_name_list = []
    for index in range(len(ref_sig_list) - 1):
    
        ref_signal = ref_sig_list[index]
        ref_signal_2 = ref_sig_list[index + 1]
        column_name = str(ref_signal) + "-" + str(ref_signal_2)
        column_name_list.append(column_name)
        
        signal_value_across_compounds = np.array([])
        for compound_value_list_of_ref_sig in value_list_of_ref_sig:
            signal_value_across_compounds = np.append(signal_value_across_compounds, compound_value_list_of_ref_sig[index])

        the_super_dictionary[column_name] = signal_value_across_compounds

    signal_df = pd.DataFrame(the_super_dictionary, index=the_super_df.index)

    scaler = MinMaxScaler()
    # let's store the index and columns
    index = signal_df.index
    columns = signal_df.columns

    # First, let's transpose the data since we want to scale it compound-wise.
    # MinMaxScaler scales columns, so to scale all the signals, each column must represent a compound,
    # with the values within the column being the signal values.
    
    signal_df = signal_df.T
    signal_df_numpy = scaler.fit_transform(signal_df)
    # Transpose again:
    signal_df_numpy = signal_df_numpy.T
    signal_df = pd.DataFrame(data=signal_df_numpy, columns=columns, index=index)

    # After the normalisation it's time for the concatenation
    the_super_df = pd.concat([the_super_df, signal_df],
                             axis="columns")
                             
    ######################
    # Sometimes the duplicates are formed !!!
    the_super_df.drop_duplicates(inplace=True)
    ######################

    # Old junk removal:
    try:
        os.remove("The_machine_proper/Uniform_csv_plain.csv")
    except:
        pass
        
    try:
        os.remove("The_machine_proper/Uniform_csv_derivative.csv")
    except:
        pass
         
    try:
        os.remove("The_machine_proper/Uniform_csv_both.csv")
    except:
        pass

    the_super_df.to_csv("The_machine_proper/Uniform_csv_plain.csv", index=False)

    #ok let's return essentially the column list of plain CSV mkay?
    return column_name_list
    
    
def Creating_Uniform_Derivative_CSV(plain_csv_signal_column_list):
    with open("Mining/Mini_dictionary.json", 'r') as openfile:
        mini_dict = json.load(openfile)
    # Since it will be used regardless of if it's categorical or not, it should be created
    the_plain_df = pd.read_csv("The_machine_proper/Uniform_csv_plain.csv")

    # Next, we'll calculate the derivative. Unfortunately, this process results in losing one column and the column names
    # becoming less intuitive. For example, the derivative from the range 800-805 to 805-810 will be named "D:800-805". Apologies for the inconvenience.
    
    derivative_dictionary = {}
    column_name_list = []
    for index in range(len(plain_csv_signal_column_list)-1):
        derivative_column_name = "D:" + plain_csv_signal_column_list[index]
        column_name_list.append(derivative_column_name)
        derivative_dictionary[derivative_column_name] = []
    
    #again going one by one compounds
    deltaX = mini_dict["Uniform_Delta_X"]
    progress_bar = Bar('Creating Uniform derivative signals: ', max=len(the_plain_df.index))
    for compound in the_plain_df.index:
        for index in range(len(plain_csv_signal_column_list)-1):
            signal_column_1 = plain_csv_signal_column_list[index]
            signal_column_2 = plain_csv_signal_column_list[index+1]

            compound_signal_value_1 = float(the_plain_df.loc[compound, signal_column_1])
            compound_signal_value_2 = float(the_plain_df.loc[compound, signal_column_2])

            derivative = float(abs( (compound_signal_value_2-compound_signal_value_1) / deltaX))
            derivative_dictionary["D:" + signal_column_1].append(derivative)
        progress_bar.next()
    progress_bar.finish()
    derivative_df = pd.DataFrame(derivative_dictionary, index=the_plain_df.index)
    #Let's make it MinMax too, so it's easier for the machine:

    scaler = MinMaxScaler()
    # let's store the index and columns
    index = derivative_df.index
    columns = derivative_df.columns
    # First let's transpose, cause we want it do scale compound-wise
    
    ########################################################################## @@@
    derivative_df = derivative_df.T
    derivative_df_numpy = scaler.fit_transform(derivative_df)
    derivative_df_numpy = derivative_df_numpy.T
    derivative_df = pd.DataFrame(data=derivative_df_numpy, index=index, columns=columns)

    derivative_df = pd.concat([derivative_df, the_plain_df[["Name", "CAS", "SMILES"]]], axis="columns")
    ########################################################################## @@@


    derivative_df.to_csv("The_machine_proper/Uniform_csv_derivative.csv", index=False)

    return column_name_list
    

    
def Make_IR_Numerical_Both():

    the_df_plain = pd.read_csv("The_machine_proper/Uniform_csv_plain.csv")
    the_df_derivative = pd.read_csv("The_machine_proper/Uniform_csv_derivative.csv")
    
    the_df = the_df_plain.merge(the_df_derivative, on=["Name", "CAS", "SMILES"], how="inner")
    the_df.reset_index(inplace=True, drop=True)
    the_df.to_csv("The_machine_proper/Uniform_csv_both.csv", index=False)
    
    return None
    

def Make_IR_Categorical(df_file, ThreshThresh, DerivativeThresh, NumericalCorrection):

    #####################################

    thresh_algorithm = False
    derivative_algorithm = False
    file_path = "The_machine_proper/" + df_file + ".csv"
    the_df = pd.read_csv(file_path)
    the_df.reset_index(inplace=True, drop=True)
    the_columns = []
    for column in the_df.columns:
        if column not in ["Name", "CAS", "SMILES"]:
            the_columns.append(column)
    special_thresh_column_names = []
    derivative_thresh_column_names = []
    thresh_thresh_column_names = []
    Anti_Noise_Only = False
    #####################################
    
    if ThreshThresh is not None and isinstance(ThreshThresh, int) is True:
        thresh_algorithm = True
        threshold_thresh = (100-float(ThreshThresh))/100


    if DerivativeThresh is not None and isinstance(DerivativeThresh, int) is True:
        derivative_algorithm = True
        derivative_thresh = float(DerivativeThresh)/100
        
    ############################################
    if DerivativeThresh == "Anti-noise-only":
        derivative_algorithm = True
        derivative_thresh = 0.05
        Anti_Noise_Only = True
    ############################################

    if thresh_algorithm is False and derivative_algorithm is False:
        thresh_algorithm = False
        derivative_algorithm = False
        print("Both Thresh are None")
        exit()

    # A 90% threshold value implies that values lower than 0.9 are passed.
    # In the case of derivatives, the opposite is true – only values greater than 0.9 are passed.
    # To avoid confusion, the threshold has been adjusted so that a higher threshold value (like in the derivative case) corresponds to a more stringent filtering.
    # Since everything has already been normalized, there is no need to create an additional column.

    if thresh_algorithm is True:
    ###############################################################################################################
        progress_bar = Bar('Making Categorical data for plain data: ', max=len(the_columns))
        progress_bar.start()
        if thresh_algorithm:
            for column in the_columns:
                new_column = []
                for index in the_df.index:

                    # The above explanation applies to plain_df. However, if both types of data are selected,
                    # then one must account for that:
                    # Derivative columns start with "D". The use of "not in []" in the previous code is
                    # for consistency and clarity.

                    if column[0] != "D" and column[0] != "C" and column not in ["Name", "CAS", "SMILES"]:
                        signal_value = float(the_df[column].loc[index])
                        
                        # For the classic threshold the point is to be lower than the threshold value
                        # noinspection PyUnboundLocalVariable
                        if signal_value < threshold_thresh:
                            new_column.append(1)
                        else:
                            new_column.append(0)
                        ##############################################    
                            
                            
                if len(new_column) != 0:
                    column_name = "Cat-" + column
                    thresh_thresh_column_names.append(column_name)
                    the_df[column_name] = new_column
                    the_df = the_df.copy()
                progress_bar.next()
            progress_bar.finish()

            print("Threshold values for thresh algorithm completed")
###############################################################################################################


###############################################################################################################

    if derivative_algorithm is True:

        #Let's get the only Anti Noise:
        if Anti_Noise_Only is True:

            special_threshold = derivative_thresh
            special_dict = {}

            #####
            progress_bar = Bar('Anti-noise correction: ', max=len(the_columns))
            progress_bar.start()
            for column in the_columns:
                if column[0] == "D" and column[0] != "C" and column not in ["Name", "CAS", "SMILES"]:
                    new_column = np.array([])
                    for index in the_df.index:
                        # Again applies to the plain one, and no C since categorical might get in the way:
                        # Yeah I know that the plain_csv is read, but let's leave it at that:
                        signal_value = float(the_df[column].loc[index])
                        
                        #The point of derivative threshold is the values to be higher than the threshold
                        #In this case the derivative needs to be higher than 0.1
                        
                        if signal_value > special_threshold:
                            new_column = np.append(new_column, 1)
                        else:
                            new_column = np.append(new_column, 0)
                                
                    if len(new_column) != 0:
                        column_name = "Cat-" + column
                        derivative_thresh_column_names.append(column_name)
                        the_df[column_name] = new_column
                        the_df = the_df.copy()
                progress_bar.next()
            progress_bar.finish()

            print("Anti-noise threshold values completed")
        ###############################################################################################################
        
        
        ###############################################################################################################
        
        else:
        
            # Here, the situation is very similar as in ThreshThresh, only this time the Derivative values are read
            progress_bar = Bar('Making Categorical data for derivative data: ', max=len(the_columns))
            progress_bar.start()
            for column in the_columns:
                if column[0] == "D" and column[0] != "C" and column not in ["Name", "CAS", "SMILES"]:
                    new_column = np.array([])
                    for index in the_df.index:
                        # Now the columns which begin with the D are accepted:
                        derivative_value = float(the_df.loc[index, column])
                        
                        #The point of derivative threshold is the values to be higher than the threshold
                        # noinspection PyUnboundLocalVariable
                        if derivative_value > derivative_thresh:
                            new_column = np.append(new_column, 1)
                        else:
                            new_column = np.append(new_column, 0)
                
                    if len(new_column) != 0:
                        column_name = "Cat-" + column
                        derivative_thresh_column_names.append(column_name)
                        the_df[column_name] = new_column
                        the_df = the_df.copy()
                progress_bar.next()
            progress_bar.finish()
            
            print("Threshold values for derivative algorithm completed")
        ###############################################################################################################


    #############################################################################################################
  
    #Ok, now the Numerical correction:
    if NumericalCorrection is True or Anti_Noise_Only is True:
        # Performing for both of thresh and derivative, so for something to retain its value
        # This sets the plain values and the derivative values to zero if the threshold is not met
        # The threshold columns are later discarded

        print("Applying the numerical correction...")
        ##########################################################################
        if len(thresh_thresh_column_names) != 0:
        
            #Original column means the column from which the categorical values were derived
        
            def fun_to_apply(dataframe):
                if dataframe[thresh_column] == 0:
                    return 1
                else:
                    return dataframe[original_column]

            #Hope it works
            for thresh_column in thresh_thresh_column_names:
                original_column = thresh_column[4:]
                the_df[original_column] = the_df.apply(fun_to_apply, axis=1)
                the_df.drop(columns=thresh_column, inplace=True)
        ##########################################################################

        ##########################################################################
        if len(derivative_thresh_column_names) != 0:
            
            #Original column means the column from which the categorical values were derived
            
            def fun_to_apply(dataframe):
                if dataframe[thresh_column] == 0:
                    return 0
                else:
                    return dataframe[original_column]

            #Hope it works
            for thresh_column in derivative_thresh_column_names:
                original_column = thresh_column[4:]
                the_df[original_column] = the_df.apply(fun_to_apply, axis=1)
                the_df.drop(columns=thresh_column, inplace=True)
        ##########################################################################
        print("Numerical correction complete")

    try:
        os.remove("The_machine_proper/Final_csv.csv")
    except:
        pass

    the_df.to_csv("The_machine_proper/Final_csv.csv", index=False)

    return None
    

def make_bonds_from_smile(file):
    import os
    import pandas as pd
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    pd.set_option('display.max_columns', None)
    
    
    with open("The_machine_proper/SMILES_SMARTS_string.json", 'r') as openfile:
        functional_group_dictionary = json.load(openfile)
    
    group_list = [key for key in functional_group_dictionary]

    path = "The_machine_proper/" + file + ".csv"
    df = pd.read_csv(path)
    df.reset_index(drop=True, inplace=True)

    #Dropping the Unnamed Smiles:

    X_column = []
    for column in df.columns:
        if column not in ["Name", "CAS", "SMILES"] and column not in group_list:
            X_column.append(column)

    how_many_one_list = []
    #one_time = 0
    #print(df)
    for key in functional_group_dictionary:
        error_to_remove = []
        functional_group_detection = []
        index = -1
        for smile_string in df["SMILES"]:
            index += 1
            matched = False
            try:
                molecule_smile_string = smile_string
                functional_group_smiles_list = functional_group_dictionary[key]

                for functional_group_smile_string in functional_group_smiles_list:
                    molecule = Chem.MolFromSmiles(molecule_smile_string)
                    try:
                        functional_group = Chem.MolFromSmarts(functional_group_smile_string)
                    except:
                        functional_group = Chem.MolFromSmiles(functional_group_smile_string)

                    match = molecule.HasSubstructMatch(functional_group)
                    if match is True:
                        matched = True
                        
                if matched is True:
                    functional_group_detection.append(1)
                else:
                    functional_group_detection.append(0)
            except:
                error_to_remove.append(index)

        for err_index in error_to_remove:
            df.drop(index=err_index, inplace=True)

        try:
            ########
            #Let's make it a minumum of 15 instances
            ########

            #################################################################
            if functional_group_detection.count(1) > 15:
            #################################################################

                df[key] = functional_group_detection
                how_many_one_list.append(functional_group_detection.count(1))
            else:
                print("Too few")
                continue
        except:
            continue

    try:
        os.remove("The_machine_proper/Final_csv.csv")
    except:
        pass

    df.to_csv("The_machine_proper/Final_csv.csv", index=False)

    ###########
    key_list = [key for key in functional_group_dictionary]
    columns = df.columns
    new_key_list = [key for key in key_list if key in columns]
    Y_column = new_key_list
    ############
   
    return X_column, Y_column, how_many_one_list

#####################################
