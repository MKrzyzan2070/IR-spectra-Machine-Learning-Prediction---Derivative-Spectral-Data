import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import numpy as np
import statistics
import math
import warnings
from itertools import combinations_with_replacement
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')


# noinspection PyRedundantParentheses
def load_the_data(method, path_empty_method, data_handling=None, resolution_range=None):

    # The current implementation is not perfect and requires attention:
    # At the moment, "PlainOnly" is being entered directly, preventing iteration.
    # In the future, it might be possible to interchange the method for "PlainOnly",
    # allowing iteration over a single method using "PlainOnly".

    if data_handling is None:
        ######################################################################################## @@@
        #Warning I used the forward backlash. Let's hope this works!!!
        path = path_empty_method + "/" + method + "_PlainOnly"
        ######################################################################################## @@@
        
    else:
        ######################################################################################## @@@
        path = path_empty_method + "/" + method + "_" + data_handling
        ######################################################################################## @@@
    
    
    all_files_list = os.listdir(path)

    #THE VARIABLES:
    #The range is from: 600 to: 3800
    #Resolution: variable - from 2.0 to 20.0
    #The state is: variable - gas, liquid, and solid
    #Plain: True
    #Derivative: False
    #Anti noise applied to derivative: False
    #Class weight: False
    #The model: SVM
    
    ######################################################################################## @@@
    sought_files_dict = {"string":[], "resolution":[], "state":[], "number":[]}
    
    ######################################################################################## @@@
    ###
    sought_string_file_1 = "Summary[\d]+_\d+_(\d+)_([a-zA-Z-]+)_[a-zA-Z-]+_[a-zA-Z-]+_[a-zA-Z-]+_[a-zA-Z-]+_[a-zA-Z-]+_[a-zA-Z-]+_[a-zA-Z-]+(\d)"
    sought_string_file_2 = "Summary[\d]+_\d+_(\d+)_([a-zA-Z-]+)_[a-zA-Z-]+_[a-zA-Z-]+_[a-zA-Z-]+[a-zA-Z-]+_[a-zA-Z-]+_[a-zA-Z-]+_[a-zA-Z-]+(\d)"
    ###
    ######################################################################################## @@@
    
    for file_string in all_files_list:
    
        the_find_1 = re.findall(sought_string_file_1, file_string)
        the_find_2 = re.findall(sought_string_file_2, file_string)
        
        if len(the_find_1) > 0:
            resolution_1 = the_find_1[0][0]
            state_1 = the_find_1[0][1]
            number_1= the_find_1[0][2]
        
        if len(the_find_2) > 0:
            resolution_2 = the_find_2[0][0]
            state_2 = the_find_2[0][1]
            number_2= the_find_2[0][2]
        
        # This part lets the user to determine which resolution is to be considered by stating the range
        # in which the resolution must find itself in
        
        if isinstance(resolution_range, tuple) is True:
            try:
                
                resolution_float = float(resolution_1)
                resolution = resolution_1
                state = state_1
                number = number_1
                
            except:
                resolution_float = float(resolution_2)
                resolution = resolution_2
                state = state_2
                number = number_2
                
            if resolution_float > resolution_range[0] and resolution_float < resolution_range[1]:
                the_file_string_to_append = file_string
                sought_files_dict["string"].append(the_file_string_to_append)
                sought_files_dict["resolution"].append(resolution)
                sought_files_dict["state"].append(state)
                sought_files_dict["number"].append(number)
        else:
            the_file_string_to_append = file_string
            sought_files_dict["string"].append(the_file_string_to_append)
            sought_files_dict["resolution"].append(resolution)
            sought_files_dict["state"].append(state)
            sought_files_dict["number"].append(number)
    ######################################################################################## @@@                               

    sought_files_df = pd.DataFrame(data=sought_files_dict)

    #######################################################################
    # Here the dictionary is created with the following structure:
    # {Moiety: list} the structure of the list is:
    # [{}, {}, {}] dict for each file. Those dics have the structure:
    # {metric: [num, num, num]}. Example:
    # {'Alkane': [{'accuracy:': ['0.8353221957040573', '0.8424821002386634',], 
    # 'precision:': ['0.8532467532467533', '0.9222797927461139'],

    moiety_dict = {}
    time_data_for_sought_files_df = {"json time":[], 
                                     "csv time":[],
                                     "training time":[]
    }
    
    for file_string in sought_files_df["string"]:
        with open(f"{path}\{file_string}") as file:
            file_lines = file.readlines()
            line_num = -1
            for line in file_lines:
                line_num += 1

                # Cheking the beginning of the moiety such as alkane or alkene:
                beginning_find = re.findall("###", line)
                if len(beginning_find) > 0:
                    #The metrics dictionary is created. It will later be added to the list of the pertaining moiety:
                    metrics_dict = {}
      
                    #Here the moiety list is created. This will be the list of metric dictionaries:
                    moiety = file_lines[line_num+1].strip("\n")
                    if moiety not in moiety_dict:
                        moiety_dict[moiety] = []

                # Getting the confusion matrix for training (also just one value each)
                # Confusion matrix looks like this:
                # The scores for train set:
                # [[2817   34]               [[ TN FP ]
                #  [  58 3779]]               [ FN TP ]]
                
                the_find = re.findall("The scores for train set:", line)
                if len(the_find) > 0:
                    #We deal with the next two lines:
                    for num in range(line_num+1, line_num+3):
                        line_of_interest = file_lines[num]
                        
                        the_find = re.findall("\d+", line_of_interest)
                        #The find looks like this: ['770', '228']
                        
                        if num == line_num+1:
                            metrics_dict["TN"] = [the_find[0]]
                            metrics_dict["FP"] = [the_find[1]]
                            
                        elif num == line_num+2:
                            metrics_dict["FN"] = [the_find[0]]
                            metrics_dict["TP"] = [the_find[1]]
                
                # Here we have looking for the lines of cross validation that look like this in the summary file:
                # accuracy:
                # [0.9641362821279139, 0.9665071770334929, 0.9599282296650717, 0.9599282296650717, 0.9659090909090909]
                # precision:
                # [0.9726890756302521, 0.974869109947644, 0.9727177334732424, 0.9704016913319239, 0.9704952581664911]

                the_find = re.findall("The Cross validation:", line)
                if len(the_find) > 0:
                    
                    #13 because we have 12 lines of metrics
                    for num in range(line_num+1, line_num+13):
                        line_of_interest = file_lines[num]
                        the_find = re.findall("\d.\d{1,}", line_of_interest)
                        if len(the_find) == 0:
                            dict_key = line_of_interest.strip("\n")
                            if dict_key not in metrics_dict:
                                metrics_dict[line_of_interest.strip("\n")] = []
                        else:
                            for metrics_number in the_find:
                                metrics_dict[dict_key].append(metrics_number)
                    moiety_dict[moiety].append(metrics_dict)
                    
                # Adding the time data that is written as the last lines in the TXT file:
                the_find = re.findall("Json preparation time: (\d+).(\d+)", line)
                if len(the_find) > 0:
                    value = float(the_find[0][0] + "." + the_find[0][1])
                    time_data_for_sought_files_df["json time"].append(value)
                
                the_find = re.findall("CSV preparation time: (\d+).(\d+)", line)
                if len(the_find) > 0:
                    value = float(the_find[0][0] + "." + the_find[0][1])
                    time_data_for_sought_files_df["csv time"].append(value)
                    
                the_find = re.findall("Working time: (\d+).(\d+)", line)
                if len(the_find) > 0:
                    value = float(the_find[0][0] + "." + the_find[0][1])
                    time_data_for_sought_files_df["training time"].append(value)
                
    additional_data_df = pd.DataFrame(data=time_data_for_sought_files_df)
    sought_files_df = pd.concat([sought_files_df, additional_data_df], axis=1)
    
    def fun_to_apply(dataframe):
        
        return dataframe["json time"] + dataframe["csv time"] + dataframe["training time"]
        
    sought_files_df["total time"] = sought_files_df.apply(fun_to_apply, axis=1)

    return (moiety_dict, sought_files_df)


# The sample of what does sought_files_df look like:

#      string resolution  state  \
#    0    Summary600_3800_10.0_gas_False_True_False_Fals...       10.0    gas
#    1    Summary600_3800_10.0_gas_False_True_False_Fals...       10.0    gas
#    2    Summary600_3800_10.0_gas_False_True_False_Fals...       10.0    gas

# The sample what does the moiety dict look like:
# {"alkane":
# [{'accuracy:': ['0.9122703023117961', '0.9075281564908121', '0.8891523414344991', '0.8956727919383521', '0.8985765124555161'],
# 'precision:': ['0.9411764705882353', '0.9432525951557094', '0.9286694101508917', '0.935127674258109', '0.9377593360995851'], '
# recall:': ['0.9568845618915159', '0.9485038274182325', '0.942240779401531', '0.942936673625609', '0.9436325678496869'],
# 'f1:': ['0.9489655172413793', '0.9458709229701596', '0.9354058721934371', '0.939015939015939', '0.9406867845993757'],
# 'roc_auc:': ['0.9478162999703962', '0.9405344467640919', '0.9287348643006264', '0.9343368128044537', '0.9313607946050032'],
# 'average_precision:': ['0.9902162262241736', '0.9870199832586093', '0.9868897406868006', '0.9868133045258604', '0.9871950771763329']},
# {'accuracy:': ['0.9116775340841731', '0.9075281564908121', '0.8885595732068761', '0.8938944872554832', '0.8979833926453143'],
# 'precision:': ['0.9417409184372858', '0.9432525951557094', '0.9280328992460589', '0.9349930843706777', '0.9365079365079365'],
# 'recall:': ['0.9554937413073713', '0.9485038274182325', '0.942240779401531', '0.9408489909

    
    
    
def make_dataframe(moiety_dict, sought_files_df):
    #######################################################################
    # Only those moieties that are present across all files !!!:
    # Yeah something has to be added to that, cause too many interesting groups are excluded !!!
    global combined_metrics_dictionary
    keys_to_delete = []
    moiety_columns = []
    for key in moiety_dict:
        number_of_files = len(sought_files_df)
        if len(moiety_dict[key]) != number_of_files:
            keys_to_delete.append(key)
        else:
            moiety_columns.append(key)

    for key in keys_to_delete:
        del moiety_dict[key]

    moiety_df = pd.DataFrame(moiety_dict)

    the_df = pd.concat([sought_files_df, moiety_df], axis=1)

    #print(the_df.loc[45, "Bromine"])
    #######################################################################

    #######################################################################
    # Now, the numbers go from 0, and the step is 5

    for moiety in moiety_columns:
        moiety_metrics_mean_list = []
        moiety_metrics_std_list = []
        moiety_combined_metrics_dict_list = []
        for index in the_df.index:
            #Initialising:
            
            #Combining the values across 5 summaries:
            ####################################################### @@@
            if index%5 == 0:
            ####################################################### @@@
            
                combined_metrics_dictionary = {}
                metrics_dictionary = the_df.loc[index, moiety]
                for metric in metrics_dictionary:
                    combined_metrics_dictionary[metric] = metrics_dictionary[metric]
            #Going forward:
            else:
                metrics_dictionary = the_df.loc[index, moiety]
                for metric in metrics_dictionary:
                    for number in metrics_dictionary[metric]:
                        combined_metrics_dictionary[metric].append(number)
            
            ####################################################### @@@
            if (index+1)%5 == 0:
            ####################################################### @@@
            
                #Now that the dictionary of five elements for a given moiety is done:
                ### The Lists:

                metric_mean_dict = {}
                metric_std_dict = {}
                for metric in combined_metrics_dictionary:
                    # five files, each of them has five values, so:
                    ### values for list:
                    mean = np.mean([float(num) for num in combined_metrics_dictionary[metric]])
                    metric_mean_dict[metric] = mean

                    std = statistics.stdev([float(num) for num in combined_metrics_dictionary[metric]])
                    
                    ################################################################################################### @@@
                    #Dividing by square root of the number of values, because it's std of a mean
                    #And times 1.959964, because it's 95%

                    metric_std_dict[metric] = 1.959964*(std/math.sqrt(len(combined_metrics_dictionary[metric])))
                    #metric_std_dict[metric] = std
                    ################################################################################################### @@@

                for i in range(5):
                    moiety_metrics_mean_list.append(metric_mean_dict)
                    moiety_metrics_std_list.append(metric_std_dict)
                    # You know what? Let's store the combined_metrics_dictionary for later use
                    moiety_combined_metrics_dict_list.append(combined_metrics_dictionary)


        #Adding the new lists to the df:
        df_column_name = f"{moiety}_mean"
        the_df[df_column_name] = moiety_metrics_mean_list

        df_column_name = f"{moiety}_std"
        the_df[df_column_name] = moiety_metrics_std_list

        df_column_name = f"{moiety}_combined_metrics"
        the_df[df_column_name] = moiety_combined_metrics_dict_list
        
        #Dropping the old column
        the_df.drop(columns=[moiety], inplace=True)

    #Now that we're done it's time to drop the duplicates:
    #Unfortuately drop_duplicates generates the error: unhashable type: 'dict', so:
    the_df = the_df[the_df["number"] == "1"]
    ########################################
    
    #Now to get rid of the need of putting ":" after each metric:
    
    # column list is ['string', 'resolution', 'state', 'number', Aromatic_mean', 'Aromatic_std']
    # Hence we take the column list of index higher than 3 that is starting from 4
    index_list = list(the_df.index)
    column_list = list(the_df.columns)[8:]
    
    for index in index_list:
        for column in column_list:
            metrics_dictionary = the_df.loc[index, column]
            
            new_metrics_dictionary = {}
            
            for metric in metrics_dictionary:
                metric_list_variable = metrics_dictionary[metric]
                the_find = re.findall(":", metric)
                if len(the_find) > 0:
                    old_metric = metric
                    new_metric = old_metric.strip(":")
                else:
                    old_metric = metric
                    new_metric = old_metric
                
                new_metrics_dictionary[new_metric] = metric_list_variable
                
            # .loc doesn't work so I had to use the .at thing
            the_df.at[index, column] = new_metrics_dictionary
                
    #{'TN': 2500.4, 'FP': 214.6, 'FN': 165.4, 'TP': 3446.6, 'accuracy:': 0.9288999525841632, 'precision:': 0.9300129483043651, 'recall:': 0.9473419898640741, 'f1:': 0.9388056406029809, 'roc_auc:': 0.9778619501080067, 'average_precision:': 0.9809290932845134}
    #{'TN': 2500.4, 'FP': 214.6, 'FN': 165.4, 'TP': 3446.6, 'accuracy': 0.9288999525841632, 'precision': 0.9300129483043651, 'recall': 0.9473419898640741, 'f1': 0.9388056406029809, 'roc_auc': 0.9778619501080067, 'average_precision': 0.9809290932845134}
    
    
    return the_df

def rearrange_columns(the_df, metric, state):

    simple_mean_dict = {}
            
    ################################################################# @@@
    #Let's do it according to the state:
    the_df_state_index = the_df.set_index("state")
    for column in the_df_state_index.columns:
        the_find = re.findall("mean",column)
        if len(the_find) > 0:
            simple_mean_dict[column] = the_df_state_index.loc[state, column][metric]
    ################################################################# @@@
    

    index_list = [moiety for moiety in simple_mean_dict]
    data_list = [simple_mean_dict[moiety] for moiety in simple_mean_dict]
    simple_mean_df = pd.DataFrame(data=data_list, index=index_list, columns=["Means"])
    simple_mean_df.sort_values(by="Means", axis=0, inplace=True, ascending=False)

    # A new order of columns needs to be created.
    # Remember the universal order: MEAN, STD, COMBINED_METRICS.
    # Also, keep in mind that the strip() method is not ideal – '_mean' has 5 letters.

    column_order_list_beta = [column[0:len(column)-5] for column in simple_mean_df.index]
    column_order_list_alfa = []

    for column_beta in column_order_list_beta:
        column_order_list_alfa.append(column_beta+"_mean")
        column_order_list_alfa.append(column_beta+"_std")
        column_order_list_alfa.append(column_beta+"_combined_metrics")

    # It's important to keep the "resolution" and the "state" 
    if "resolution" in the_df.columns:
        resolution_column = the_df["resolution"]
    else:
        resolution_column = None
        
    if "state" in the_df.columns:
        state_column = the_df["state"]
    else:
        state_column = None

    # And it's time for the final rearrangement:
    the_df = the_df[column_order_list_alfa]
    
    if resolution_column is not None:
        the_df["resolution"] = resolution_column
    if state_column is not None:
        the_df["state"] = state_column
    
    return the_df


def make_axis_list(the_df, column, metric_list):
    x_axis = []
    x_axis_string = []
    y_axis = []
    hue = []
    column_found = True
    for index in the_df.index:
        hue_value = the_df.loc[index, "state"]
        x_value = -1
        for metric in the_df.loc[index, column]:
            if metric_list is None or metric in metric_list:
                x_value += 1
                x_string = metric
                y_value = the_df.loc[index, column][metric]
                if isinstance(y_value, list):
                    for number in y_value:
                        y_axis.append(float(number))
                        x_axis.append(x_value)
                        x_axis_string.append(x_string)
                        hue.append(hue_value)
                else:
                    y_axis.append(float(y_value))
                    x_axis.append(x_value)
                    x_axis_string.append(x_string)
                    hue.append(hue_value)
                    
    return (x_axis, x_axis_string, y_axis, hue, column_found)





def make_std_list(the_df, column, metric_list):
    std_list = []
    std_found = True
    for index in the_df.index:
        if metric_list is None:
            for metric in the_df.loc[index, column]:
                std_value = the_df.loc[index, column][metric]
                std_list.append(float(std_value))
        else:
            for metric in the_df.loc[index, column]:
                if metric in metric_list:
                    std_value = the_df.loc[index, column][metric]
                    std_list.append(float(std_value))

    return (std_list, std_found)



def make_error_bar(subplot, the_df, column, hue, x_axis, y_axis, std_list, width, plot_type, grid_place_1, grid_place_2):

    index = the_df.index
    
    numbers_of_bars_in_group = len(the_df)
    error_x_axis_list_of_lists = [[] for i in range(numbers_of_bars_in_group)]
    error_y_axis_list_of_lists = [[] for i in range(numbers_of_bars_in_group)]
    error_value_list_of_lists = [[] for i in range(numbers_of_bars_in_group)]
    true_order_num = 0
    order_num = 0
    val_to_add_to_x_number_list = list(np.linspace(-(numbers_of_bars_in_group-1)*width/2, (numbers_of_bars_in_group-1)*width/2, 
                                 numbers_of_bars_in_group))

    for x_value in x_axis:

        true_order_num += 1
        if x_value == 0:
            order_num += 1

        std_value = std_list[true_order_num-1]
        y_value = y_axis[true_order_num-1]

        for in_group_order_num in range(numbers_of_bars_in_group, 0, -1):

            if order_num == in_group_order_num:
                val_to_add_to_x_number = val_to_add_to_x_number_list[in_group_order_num-1]

                error_x_axis_list_of_lists[in_group_order_num-1].append(x_value+val_to_add_to_x_number)
                error_y_axis_list_of_lists[in_group_order_num-1].append(y_value)
                error_value_list_of_lists[in_group_order_num-1].append(std_value)
                number_in_group_found = True
                break

    
    error_index = -1
    for error_x_axis_list in error_x_axis_list_of_lists:
        error_index += 1
        error_value_list = error_value_list_of_lists[error_index]
        error_y_axis_list = error_y_axis_list_of_lists[error_index]

        if plot_type == "bar":
            subplot.errorbar(x=error_x_axis_list, y=error_y_axis_list, yerr=error_value_list,
                                                      color="black", fmt="D", elinewidth=4.0, capsize= 16.0, markersize=10.0, zorder=100)

        else:
            subplot.errorbar(x=error_x_axis_list, y=error_y_axis_list, yerr=error_value_list,
                                                      color="black", fmt='D', elinewidth=4.0, capsize= 16.0, markersize=18.0, zorder=100)
            
            
    return None


def get_subplot_dimensions(figure_plot_list):

    subplots_number = len(figure_plot_list)
    
    if subplots_number/2 != round(subplots_number/2):
        subplots_number_new = subplots_number + 1
    else:
        subplots_number_new = subplots_number
        
    numbers = np.arange(subplots_number_new, 0, -1)
    highest_num = numbers[0]
    potential_numbers = []

    for number in numbers:
        potential_numbers.append(number)
            
    #First number is always going to be higher:
    potential_pair = (highest_num, 1)
    for num_pair in combinations_with_replacement(potential_numbers, 2):
        num1 = num_pair[0]
        num2 = num_pair[1]
        if num1*num2 == highest_num:
            potential_pair_diff = potential_pair[0] - potential_pair[1]
            if num1-num2 <= potential_pair_diff:
                potential_pair = (num1, num2)

    return (potential_pair, subplots_number)    
    
def make_scores_across_moieties_plot(the_df, moiety_list):
    # It's time to make a function that displays class imbalance, and the mean value of f1 only
    sns.set_theme(style="whitegrid", palette="muted")

    x_string_list = []
    x_list = []
    y_data_dict = {}
    i=-1
    for column in the_df.columns:
        
        plot_data = {}
        ####################################################
        # In case the moiety_list is provided:
        moiety_name = column.split("_")[0]

        # Small renaming
        if moiety_name == "Non-aromatic Halide":
            moiety_name = "Non-aromatic Halogen"
            
        elif moiety_name == "Aromatic halogen":
            moiety_name = "Aromatic Halogen"
            
        elif moiety_name == "Aromatic nitrogen":
            moiety_name = "Aromatic Nitrogen"
            
        elif moiety_name == "Aromatic ether":
            moiety_name = "Aromatic Ether"
        
        
        # Skipping
        if moiety_list is not None:
            if moiety_name not in moiety_list:
                continue
        ####################################################

        # Making the axis, which the mean !!!:
        if len(re.findall("mean", column)) > 0:
            
            i += 1
            x_string_list.append(moiety_name)
            x_list.append(i)
            # We need to get the TP, TN, FP, and FN. And of course the f1 score
            
            # The values will calculate as the mean of all the values across the physical state
            for index in the_df.index:
                the_dictionary = the_df.loc[index, column]
                
                if index == 0:
                    #First definition of the dictionary keys --> making lists
                    plot_data["TP"] = [the_dictionary["TP"]]
                    plot_data["TN"] = [the_dictionary["TN"]]
                    plot_data["FP"] = [the_dictionary["FP"]]
                    plot_data["FN"] = [the_dictionary["FN"]]
                    plot_data["f1"] = [the_dictionary["f1"]]
                    plot_data["accuracy"] = [the_dictionary["accuracy"]]
                    plot_data["average_precision"] = [the_dictionary["average_precision"]]
                    
                else:
                    #Appending to a list
                    plot_data["TP"].append(the_dictionary["TP"])
                    plot_data["TN"].append(the_dictionary["TN"])
                    plot_data["FP"].append(the_dictionary["FP"])
                    plot_data["FN"].append(the_dictionary["FN"])
                    plot_data["f1"].append(the_dictionary["f1"])
                    plot_data["accuracy"].append(the_dictionary["accuracy"])
                    plot_data["average_precision"].append(the_dictionary["average_precision"])

        # Now that we have the data, let's make the mean value for each and add the name of the columm = moiety_name
        for key in plot_data:
            the_list = plot_data[key]
            the_mean = np.mean(the_list)
            
            # Declaring the keys for the y_data_dict
            if i == 0:
                y_data_dict[key] = [the_mean]
                
            # Adding on the built lists
            else:
                y_data_dict[key].append(the_mean)
                
    #Now to calculate the imbalance ratio:
    y_data_df = pd.DataFrame(y_data_dict)

    x_data_dict = {"x_list": x_list, "x_string_list": x_string_list}
    x_data_df = pd.DataFrame(x_data_dict)

    # Axis=1 means row, Axis=0 means column
    def fun_to_apply(row):
        positives = row["TP"] + row["FN"]
        negatives = row["FP"] + row["TN"]
        imbalance_ratio = positives/negatives
        if imbalance_ratio > 1:
            imbalance_ratio = 1/imbalance_ratio
            
        return imbalance_ratio

    y_data_df["Imbalance ratio"] = y_data_df.apply(fun_to_apply, axis=1)
    y_data_df = y_data_df[["f1", "accuracy", "Imbalance ratio", "average_precision"]]
    y_data_df = y_data_df.sort_values("Imbalance ratio", ascending=False)
    x_data_df = x_data_df.loc[list(y_data_df.index)]

    x_data_df.reset_index(inplace=True, drop=True)
    y_data_df.reset_index(inplace=True, drop=True)
    x_data_df["x_list"] = list(x_data_df.index)

    # Purple rgb - rgb(138,43,226)
    # Dark green - rgb(0,200,0) 
    # Yellow - rgb(204,204,0)
    figure = plt.figure(figsize=(23, 6), dpi=100)

    # First let's make the imbalance plot
    purple_color = (138/255, 43/255, 226/255)
    #Creating axes
    plot1 = figure.add_subplot(111)
    plot1.plot(list(x_data_df["x_list"]), list(y_data_df["Imbalance ratio"]), linewidth=4, color=purple_color, label="imbalance ratio")
    plot1.set_xticks(x_data_df["x_list"])
    plot1.set_xticklabels(x_data_df["x_string_list"], size=16, rotation=45)
    y_axis_nums = [float(num) for num in range(0,12,2)]
    y_axis_nums = [num/10 for num in y_axis_nums]
    y_axis_string = [str(num) for num in y_axis_nums]
    plot1.set_yticks(y_axis_nums)
    plot1.set_yticklabels(y_axis_string, size=16)
    plot1.set_ylim((0, 1))
    plot1.set_xlabel("organic moiety", size=26)
    plot1.set_ylabel("score value", size=26)

    #Now the second plot that shares the x_axis:
    dark_green_color = (0, 200/255, 0)
    plot1.plot(list(x_data_df["x_list"]), list(y_data_df["f1"]), linewidth=4, color=dark_green_color, label="f1 score")

    #Maybe let's alo add the acuracy score:
    yellow_color = (204/255,204/255,0)
    plot1.plot(list(x_data_df["x_list"]), list(y_data_df["average_precision"]), linewidth=4, color=yellow_color, label="average precision score")

    plt.legend(loc='center right', fontsize=24)
    
    plt.title("KNN; gas state", fontsize=28)

    plt.show()

    return None


def make_strip_plot(the_df, metric_list, moiety_list=None):

    global mean_x_axis, column, hue, y_axis, x_axis_string, x_axis, std_list, mean_hue, mean_y_axis, mean_x_axis_string
    sns.set_theme(style="whitegrid", palette="bright")
    
    mean_found = False
    std_found = False
    combined_metrics_found = False

    ##################################################
    # CREATING THE FIGURE_PLOT_LIST THAT CONTAINES THE DATA FOR THE GIVEN SUBPLOT:
    ##################################################
        
    figure_plot_list = []
    subplot_data = {}
    
    for column in the_df.columns:
        
        # cause it sometimes glitches: 
        index = the_df.index
        
        ####################################################
        # In case the moiety_list is provided:
        moiety_name = column.split("_")[0]
        
        # Small renaming
        if moiety_name == "Non-aromatic Halide":
            moiety_name = "Non-aromatic Halogen"
            
        elif moiety_name == "Aromatic halogen":
            moiety_name = "Aromatic Halogen"
            
        elif moiety_name == "Aromatic nitrogen":
            moiety_name = "Aromatic Nitrogen"
            
        elif moiety_name == "Aromatic ether":
            moiety_name = "Aromatic Ether"
        
        # Skipping
        if moiety_list is not None:
            if moiety_name not in moiety_list:
                continue
                
        ####################################################
        
        # IT IS VERY IMPORTANT TO KEEP THE ORDER: MEAN, STD, COMBINED_METRICS
        if len(re.findall("mean", column)) > 0:
            (mean_x_axis, mean_x_axis_string, mean_y_axis, mean_hue, mean_found) = make_axis_list(the_df, column, metric_list)  

        #making std: 
        if len(re.findall("std", column)) > 0:
            (std_list, std_found) = make_std_list(the_df, column, metric_list)
            
        #making the axis:
        if len(re.findall("combined_metrics", column)) > 0:
            (x_axis, x_axis_string, y_axis, hue, combined_metrics_found) = make_axis_list(the_df, column, metric_list)

        
        if combined_metrics_found is True and mean_found is True and std_found is True:
        
            subplot_data["mean_x_axis"] = mean_x_axis
            subplot_data["mean_x_axis_string"] = mean_x_axis_string
            subplot_data["mean_y_axis"] = mean_y_axis
            subplot_data["mean_hue"] = mean_hue
            subplot_data["std_list"] = std_list
            subplot_data["x_axis"] = x_axis
            subplot_data["x_axis_string"] = x_axis_string
            subplot_data["y_axis"] = y_axis
            subplot_data["hue"] = hue
            
            #Lastly the title of the subplot
            title = column.split("_")[0]
            if title == "ColdMLP":
                title = "MLP"
                
            elif title == "Non-aromatic Halide":
                title = "Non-aromatic Halogen"
                
            elif title == "Aromatic halogen":
                title = "Aromatic Halogen"
                
            elif title == "Aromatic nitrogen":
                title = "Aromatic Nitrogen"
                
            elif title == "Aromatic ether":
                title = "Aromatic Ether"
                
            subplot_data["title"] = title
        
            figure_plot_list.append(subplot_data)
            subplot_data = {}
            combined_metrics_found = False
            mean_found = False
            std_found = False


    ##################################################
    # Plotting
    ##################################################
    
    # Before doing anything:
    output = get_subplot_dimensions(figure_plot_list)
    
    dimensions = output[0]
    subplots_number = output[1]
    
    #Creating figure and plot:
    figure, plot = plt.subplots(dimensions[0], dimensions[1], figsize=(dimensions[1]*18, dimensions[0]*14))
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
        
    grid_place_1 = -1
    grid_place_2 = 0
    
    for i in range(subplots_number):
    
        grid_place_1 += 1
        if grid_place_1 >= dimensions[0]:
            grid_place_1 = 0
            grid_place_2 += 1

        ######## Getting data: #######
        
        #First thing is to get the data for currently created subplot:
        subplot_data = figure_plot_list[i]
        
        #And there we go:
        mean_x_axis = subplot_data["mean_x_axis"]
        mean_x_axis_string = subplot_data["mean_x_axis_string"]
        mean_y_axis = subplot_data["mean_y_axis"]
        mean_hue = subplot_data["mean_hue"]
        std_list = subplot_data["std_list"]
        x_axis = subplot_data["x_axis"]
        x_axis_string = subplot_data["x_axis_string"]
        y_axis = subplot_data["y_axis"]
        hue = subplot_data["hue"]
        title = subplot_data["title"]
        if title == "ColdMLP":
                title = "MLP"
        
        #Now the proper plotting:
        
        #For two-dimensional instances:
        try:
            subplot = sns.stripplot(ax=plot[grid_place_1, grid_place_2], x=x_axis, y=y_axis, hue=hue, size=16, dodge=True, jitter=True)
            ######################################################################################## @@@
            width = 0.8/3
            ######################################################################################## @@@
            make_error_bar(subplot, the_df, column, mean_hue, mean_x_axis, mean_y_axis, std_list, width, 
                           "strip", grid_place_1, grid_place_2)
           
            #Final touch:
            plot[grid_place_1, grid_place_2].set_ylim([0.0, 1.0])
            plot[grid_place_1, grid_place_2].set_title(title, fontsize=42)
            #I am sorry this has to be done:
            x_axis_string = list(dict.fromkeys(x_axis_string))
            plot[grid_place_1, grid_place_2].set_xticklabels(x_axis_string, fontsize=32)
            plot[grid_place_1, grid_place_2].set_xlabel("metric", fontsize=38)
            
            # I have to add this tick label nonsence :(
            y_axis_string = [float(num) for num in range(0,12,2)]
            y_axis_string = [str(num/10) for num in y_axis_string]
            plot[grid_place_1, grid_place_2].set_yticklabels(y_axis_string ,fontsize=32)
            plot[grid_place_1, grid_place_2].legend(loc='center right', bbox_to_anchor=(1.15, 0.5), fontsize=38, markerscale=3)
            plot[grid_place_1, grid_place_2].set_ylabel("metric score value", fontsize=38)
            
           
            
        #For one-dimensional instances
        except:
            subplot = sns.stripplot(ax=plot[grid_place_1], x=x_axis, y=y_axis, hue=hue, size=16, dodge=True, jitter=True)
            ######################################################################################## @@@
            width = 0.8/3
            ######################################################################################## @@@
            make_error_bar(subplot, the_df, column, mean_hue, mean_x_axis, mean_y_axis, std_list, width, 
                           "strip", grid_place_1, grid_place_2)
           
            #Final touch:
            plot[grid_place_1].set_ylim([0.0, 1.0])
            plot[grid_place_1].set_title(title, fontsize=42)
            #I am sorry this has to be done:
            x_axis_string = list(dict.fromkeys(x_axis_string))
            plot[grid_place_1].set_xticklabels(x_axis_string, fontsize=32)
            plot[grid_place_1, grid_place_2].set_xlabel("metric", fontsize=38)
            
            # I have to add this tick label nonsence :(
            y_axis_string = [float(num) for num in range(0,12,2)]
            y_axis_string = [str(num/10) for num in y_axis_string]
            plot[grid_place_1].set_yticklabels(y_axis_string ,fontsize=32)
            plot[grid_place_1].legend(loc='center right', bbox_to_anchor=(1.15, 0.5), fontsize=38, markerscale=3)
            plot[grid_place_1, grid_place_2].set_ylabel("metric score value", fontsize=38)
        
    return None
            
            
            
def make_bar_plot(the_df, metric_list, moiety_list=None):
   
   
    global column, std_list, mean_hue, mean_y_axis, mean_x_axis_string, mean_x_axis
    sns.set_theme(style="whitegrid", palette="muted")
    

    ##################################################
    # CREATING THE FIGURE_PLOT_LIST THAT CONTAINES THE DATA FOR THE GIVEN SUBPLOT:
    ##################################################
        
    mean_found = False
    std_found = False
    figure_plot_list = []
    subplot_data = {}

    for column in the_df.columns:

        # cause it sometimes glitches: 
        index = the_df.index
        
        ####################################################
        # In case the moiety_list is provided:
        moiety_name = column.split("_")[0]
        
        
        # Small renaming
        if moiety_name == "Non-aromatic Halide":
            moiety_name = "Non-aromatic Halogen"
            
        elif moiety_name == "Aromatic halogen":
            moiety_name = "Aromatic Halogen"
            
        elif moiety_name == "Aromatic nitrogen":
            moiety_name = "Aromatic Nitrogen"
            
        elif moiety_name == "Aromatic ether":
            moiety_name = "Aromatic Ether"
            
            
        # Skipping
        if moiety_list is not None:
            if moiety_name not in moiety_list:
                continue
        ####################################################

        # IT IS VERY IMPORTANT TO KEEP THE ORDER: MEAN, STD, COMBINED_METRICS

        # Making the axis, which the mean !!!:
        if len(re.findall("mean", column)) > 0:
            (mean_x_axis, mean_x_axis_string, mean_y_axis, mean_hue, mean_found) = make_axis_list(the_df, column, metric_list)  

        # Making std: 
        if len(re.findall("std", column)) > 0:
            (std_list, std_found) = make_std_list(the_df, column, metric_list)
    

        if mean_found is True and std_found is True:

            subplot_data["mean_x_axis"] = mean_x_axis
            subplot_data["mean_x_axis_string"] = mean_x_axis_string
            subplot_data["mean_y_axis"] = mean_y_axis
            subplot_data["mean_hue"] = mean_hue
            subplot_data["std_list"] = std_list

            #Lastly the title of the subplot
            title = column.split("_")[0]
            if title == "ColdMLP":
                title = "MLP"
                
            elif title == "Non-aromatic Halide":
                title = "Non-aromatic Halogen"
                
            elif title == "Aromatic halogen":
                title = "Aromatic Halogen"
                
            elif title == "Aromatic nitrogen":
                title = "Aromatic Nitrogen"
                
            elif title == "Aromatic ether":
                title = "Aromatic Ether"
                
            subplot_data["title"] = title

            figure_plot_list.append(subplot_data)
            subplot_data = {}
            mean_found = False
            std_found = False


    ##################################################
    # Plotting
    ##################################################
    
    # Before doing anything:
    output = get_subplot_dimensions(figure_plot_list)
    
    dimensions = output[0]
    subplots_number = output[1]
    
    #Creating figure and plot:
    figure, plot = plt.subplots(dimensions[0], dimensions[1], figsize=(dimensions[1]*18, dimensions[0]*14))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    grid_place_1 = -1
    grid_place_2 = 0
    
    for i in range(subplots_number):
    
        grid_place_1 += 1
        if grid_place_1 >= dimensions[0]:
            grid_place_1 = 0
            grid_place_2 += 1

        ######## Getting data: #######
        
        #First thing is to get the data for currently created subplot:
        subplot_data = figure_plot_list[i]
        
        #And there we go:
        mean_x_axis = subplot_data["mean_x_axis"]
        mean_x_axis_string = subplot_data["mean_x_axis_string"]
        mean_y_axis = subplot_data["mean_y_axis"]
        mean_hue = subplot_data["mean_hue"]
        std_list = subplot_data["std_list"]
        title = subplot_data["title"]
        if title == "ColdMLP":
            title = "MLP"
        
        #Now the proper plotting:
        
        #For two-dimensional instances:
        try:
            subplot = sns.barplot(ax=plot[grid_place_1, grid_place_2], x=mean_x_axis, y=mean_y_axis, hue=mean_hue)
            ######################################################################################## @@@
            width = 0.8/3
            ######################################################################################## @@@
            make_error_bar(subplot, the_df, column, mean_hue, mean_x_axis, mean_y_axis, std_list, width, "bar", 
                   grid_place_1, grid_place_2)
               
            #Final touch:
            plot[grid_place_1, grid_place_2].set_ylim([0.0, 1.0])
            plot[grid_place_1, grid_place_2].set_title(title, fontsize=42)
            #I am sorry this has to be done:
            mean_x_axis_string = list(dict.fromkeys(mean_x_axis_string))
            plot[grid_place_1, grid_place_2].set_xticklabels(mean_x_axis_string, fontsize=32)
            plot[grid_place_1, grid_place_2].set_xlabel("metric", fontsize=38)
            
            # I have to add this tick label nonsence :(
            y_axis_string = [float(num) for num in range(0,12,2)]
            y_axis_string = [str(num/10) for num in y_axis_string]
            plot[grid_place_1, grid_place_2].set_yticklabels(y_axis_string ,fontsize=32)
            plot[grid_place_1, grid_place_2].legend(loc='center right', bbox_to_anchor=(1.15, 0.5), fontsize=38)
            plot[grid_place_1, grid_place_2].set_ylabel("metric score value", fontsize=38)
                
                
            
        #For one-dimensional instances:
        except:
            subplot = sns.barplot(ax=plot[grid_place_1], x=mean_x_axis, y=mean_y_axis, hue=mean_hue)
            ######################################################################################## @@@
            width = 0.8/3
            ######################################################################################## @@@
            make_error_bar(subplot, the_df, column, mean_hue, mean_x_axis, mean_y_axis, std_list, width, "bar", 
                   grid_place_1, grid_place_2)
                   
            #Final touch:
            plot[grid_place_1].set_ylim([0.0, 1.0])
            plot[grid_place_1].set_title(title, fontsize=42)
            #I am sorry this has to be done:
            mean_x_axis_string = list(dict.fromkeys(mean_x_axis_string))
            plot[grid_place_1].set_xticklabels(mean_x_axis_string, fontsize=32)
            plot[grid_place_1, grid_place_2].set_xlabel("metric", fontsize=38)
            
            # I have to add this tick label nonsence :(
            y_axis_string = [float(num) for num in range(0,12,2)]
            y_axis_string = [str(num/10) for num in y_axis_string]
            plot[grid_place_1].set_yticklabels(y_axis_string ,fontsize=32)
            plot[grid_place_1].legend(loc='center right', bbox_to_anchor=(1.15, 0.5), fontsize=38)
            plot[grid_place_1, grid_place_2].set_ylabel("metric score value", fontsize=38)

    return None


def make_resolution_line_plot_data_making(the_df, state_string, method, metric_list, reverse, doplot, resolution_range=None):
    
    global mean_y_axis, mean_hue, y_axis, hue
    resolution_list = list(set([float(resolution) for resolution in the_df["resolution"]]))

    hue_resolution_across_resolution = []
    y_axis_across_resolution = []
    x_axis_moiety_across_resolution = []
    x_axis_name_moiety_across_resolution = []

    hue_mean_resolution_across_resolution = []
    y_axis_mean_across_resolution = []
    x_axis_mean_moiety_across_resolution = []
    x_axis_mean_name_moiety_across_resolution = []
    
    total_time_across_resolution = []
    csv_time_across_resolution = []
    json_time_across_resolution = []
    training_time_across_resolution = []

    anti_loop = False
    
    the_df["resolution"] = [int(res) for res in list(the_df["resolution"])]

    for resolution in resolution_list:

        if not anti_loop:
            #anti_loop=True

            #Creating the new dataframe that has only the given resolution
            the_df_given_res = the_df[the_df["resolution"] == int(resolution)]
            the_df_given_res.reset_index(inplace=True)
            the_df_given_res.drop(columns=["index"], inplace=True)
            
            #######################################################################
            the_df_given_res_rearranged = rearrange_columns(the_df_given_res, metric_list[0], state_string)
            for initial_column in list(the_df_given_res.columns):
                if initial_column not in list(the_df_given_res_rearranged.columns):
                    the_df_given_res_rearranged[initial_column] = the_df_given_res[initial_column]
            the_df_given_res = the_df_given_res_rearranged
            #######################################################################
            
            the_df_given_res.reset_index(inplace=True)
            the_df_given_res.drop(columns=["index"], inplace=True)

            mean_found = False
            combined_metrics_found = False

            x_axis_moiety_value = -1
            for column in the_df_given_res.columns:
                index = the_df_given_res.index

                # IT IS VERY IMPORTANT TO KEEP THE ORDER: MEAN, STD, COMBINED_METRICS
                # the_mean:
                if len(re.findall("mean", column)) > 0:
                    (mean_x_axis, mean_x_axis_string, mean_y_axis, mean_hue, mean_found) = make_axis_list(the_df_given_res
                                                                                                          , column, metric_list)  

                # all the points if nedded:
                if len(re.findall("combined_metrics", column)) > 0:
                    (x_axis, x_axis_string, y_axis, hue, combined_metrics_found) = make_axis_list(the_df_given_res
                                                                                                  , column, metric_list)

                if combined_metrics_found is True and mean_found is True:
                    x_axis_moiety_value += 1

                    ####################################################################################
                    #Here comes the regulars:
                    correct_index_list = []
                    index = -1
                    for state in hue:
                        index += 1
                        ##################################################### @@@
                        if state == state_string: 
                            correct_index_list.append(index)
                        ##################################################### @@@

                    y_axis_new = [y_axis[index] for index in correct_index_list]
                    y_axis_across_resolution.append(y_axis_new)

                    for i in range(len(y_axis_new)):
                        x_axis_moiety_across_resolution.append(x_axis_moiety_value)
                        x_axis_name_moiety_across_resolution.append(column)
                        hue_resolution_across_resolution.append(resolution)
                    ####################################################################################

                    ####################################################################################
                    #Here comes the mean:
                    correct_index = 0
                    index = -1
                    for state in mean_hue:
                        index += 1
                        ##################################################### @@@
                        if state == state_string: 
                            correct_index = index
                        ##################################################### @@@

                    y_axis_mean_new = mean_y_axis[correct_index]
                    y_axis_mean_across_resolution.append(y_axis_mean_new)
                    
                    # And finally the_time
                    # dataframe["json time"] + dataframe["csv time"] + dataframe["training time"]
                    
                    # total_time
                    total_time_for_states = list(the_df_given_res["total time"])
                    total_time_singular_state = total_time_for_states[correct_index]
                    total_time_across_resolution.append(total_time_singular_state)
                    
                    # csv time
                    csv_time_for_states = list(the_df_given_res["csv time"])
                    csv_time_singular_state = csv_time_for_states[correct_index]
                    csv_time_across_resolution.append(csv_time_singular_state)
                    
                    #json time
                    json_time_for_states = list(the_df_given_res["json time"])
                    json_time_singular_state = json_time_for_states[correct_index]
                    json_time_across_resolution.append(json_time_singular_state)
                    
                    #training time
                    training_time_for_states = list(the_df_given_res["training time"])
                    training_time_singular_state = training_time_for_states[correct_index]
                    training_time_across_resolution.append(training_time_singular_state)
                    
                    
                    for i in range(1):
                        x_axis_mean_moiety_across_resolution.append(x_axis_moiety_value)
                        x_axis_mean_name_moiety_across_resolution.append(column)
                        hue_mean_resolution_across_resolution.append(resolution)


                    mean_found = False
                    combined_metrics_found = False

    # The length of hue_resolution_across_resolution, y_axis_across_resolution,
    # y_axis_mean_across_resolution, x_axis_moiety_across_resolution, and
    # x_axis_name_moiety_across_resolution for each resolution is 14, as there are 14 moieties.
    # Let's create a graph of just the mean:
    # In this case, each resolution has only one x_value and one y_value, since it's just the mean.
    # The sort method ensures that the resolution number ranges from the lowest to the highest.

    the_hue = [int(num) for num in hue_resolution_across_resolution]
    hue_resolution_across_resolution = the_hue
    hue_resolution_across_resolution.sort()
    #Strippig doesn't work so: "_combined_metrics" is the last 17 letters
    string_stripping = [string[0:len(string)-17] for string in x_axis_mean_name_moiety_across_resolution]
    x_axis_mean_name_moiety_across_resolution = string_stripping
    
    if reverse is False:
        if doplot is True:
            figure = plt.figure(figsize=(20,6))
            
            plot = sns.lineplot(x=x_axis_mean_moiety_across_resolution ,y=y_axis_mean_across_resolution, 
                                hue=hue_mean_resolution_across_resolution, palette="magma", legend="full", linewidth=2.5)
            plot.set_xticks(x_axis_mean_moiety_across_resolution)
            plt.legend(loc="center right", fontsize=14, markerscale=2)
            
            ##############################################
            new_list = []
            for name in x_axis_mean_name_moiety_across_resolution:
                if name == "Carbonyl group. Low specificity":
                    name = "Carbonyl (any)"
                    new_list.append(name)
                elif name == "Primary or secondary amine":
                    name = "Amine (any)"
                    new_list.append(name)
                    
                elif name == "Non-aromatic Halide":
                    name = "Non-aromatic Halogen"
                    new_list.append(name)
                
                elif name == "Aromatic halogen":
                    name = "Aromatic Halogen"
                    new_list.append(name)
                
                elif name == "Aromatic nitrogen":
                    name = "Aromatic Nitrogen"
                    new_list.append(name)
                
                elif name == "Aromatic ether":
                    name = "Aromatic Ether"
                    new_list.append(name)
                    
                else:
                    new_list.append(name)
                    
            x_axis_mean_name_moiety_across_resolution = new_list
            ##############################################    
                
            plot.set_xticklabels(x_axis_mean_name_moiety_across_resolution, rotation=45, size=14)
            plot.set_ylim(0.0, 1.0)
            plot.set_xlim(x_axis_mean_moiety_across_resolution[0], x_axis_mean_moiety_across_resolution[-1])
            
        else:
        
            ##############################################
            new_list = []
            for name in x_axis_mean_name_moiety_across_resolution:
                if name == "Carbonyl group. Low specificity":
                    name = "Carbonyl (any)"
                    new_list.append(name)
                    
                elif name == "Primary or secondary amine":
                    name = "Amine (any)"
                    new_list.append(name)
                    
                elif name == "Non-aromatic Halide":
                    name = "Non-aromatic Halogen"
                    new_list.append(name)
                
                elif name == "Aromatic halogen":
                    name = "Aromatic Halogen"
                    new_list.append(name)
                
                elif name == "Aromatic nitrogen":
                    name = "Aromatic Nitrogen"
                    new_list.append(name)
                
                elif name == "Aromatic ether":
                    name = "Aromatic Ether"
                    new_list.append(name)
                    
                else:
                    new_list.append(name)
                    
            x_axis_mean_name_moiety_across_resolution = new_list
            ##############################################
        
            
            the_dict_to_return = {"x": x_axis_mean_moiety_across_resolution,
                                  "y": y_axis_mean_across_resolution,
                                  "hue": hue_mean_resolution_across_resolution,
                                  "x_ticks": x_axis_mean_moiety_across_resolution,
                                  "x_ticks_labels": x_axis_mean_name_moiety_across_resolution,
                                  "palette": "magma",
                                  "legend": "full",
                                  "linewidth": 2.5,
                                  "total time": total_time_across_resolution, 
                                  "csv time": csv_time_across_resolution,
                                  "json time": json_time_across_resolution,
                                  "training time": training_time_across_resolution                                  
            }
            
            return the_dict_to_return
            
        
    else:
    
        if doplot is True:
            figure = plt.figure(figsize=(20,6))
            
            ##############################################
            new_list = []
            for name in x_axis_mean_name_moiety_across_resolution:
                if name == "Carbonyl group. Low specificity":
                    name = "Carbonyl (any)"
                    new_list.append(name)
                elif name == "Primary or secondary amine":
                    name = "Amine (any)"
                    new_list.append(name)
                else:
                    new_list.append(name)
            
            x_axis_mean_name_moiety_across_resolution = new_list
            hue_mean_name_resolution_across_resolution = []
            for hue in hue_mean_resolution_across_resolution:
                if hue not in hue_mean_name_resolution_across_resolution:
                    hue_mean_name_resolution_across_resolution.append(hue)
            ##############################################
            
            
            plot = sns.lineplot(x=hue_mean_resolution_across_resolution ,y=y_axis_mean_across_resolution, 
                                hue=x_axis_mean_name_moiety_across_resolution, palette="Spectral", legend="full", linewidth=2.5)
            plt.legend(loc="center right", fontsize=14, markerscale=2)
            plot.set_xticks(hue_mean_name_resolution_across_resolution)
            plot.set_xticklabels(hue_mean_name_resolution_across_resolution, rotation=45, size=14)
            
            # The lower value is the 20% of the difference between the upper value - always 1.0 and the minimum of y_axis
            y_minimum = min(y_axis_mean_across_resolution)
            lower_limit = y_minimum - 0.2*(1-y_minimum)
            if lower_limit < 0:
                lower_limit = 0
            
            plot.set_ylim(lower_limit, 1.0)
            plt.yticks(size=14)
            plot.set_xlim(hue_mean_resolution_across_resolution[0], hue_mean_resolution_across_resolution[-1])
            
            
        else:
            
            ### Here the more standard thing is being done. The dictionary is created is created 
            ### upon which later plot of subplots will be created
        
            ###########################################
            new_list = []
            
            ########## Changing some names
            for name in x_axis_mean_name_moiety_across_resolution:
                if name == "Carbonyl group. Low specificity":
                    name = "Carbonyl (any)"
                    new_list.append(name)
                elif name == "Primary or secondary amine":
                    name = "Amine (any)"
                    new_list.append(name)
                else:
                    new_list.append(name)
            ##########
            
            x_axis_mean_name_moiety_across_resolution = new_list
            hue_mean_name_resolution_across_resolution = []
            for hue in hue_mean_resolution_across_resolution:
                if hue not in hue_mean_name_resolution_across_resolution:
                    hue_mean_name_resolution_across_resolution.append(hue)
                    
            ###########################################
        
            the_dict_to_return = {"x": hue_mean_resolution_across_resolution,
                                  "y": y_axis_mean_across_resolution,
                                  "hue": x_axis_mean_name_moiety_across_resolution,
                                  "x_ticks": hue_mean_name_resolution_across_resolution,
                                  "x_ticks_labels": hue_mean_name_resolution_across_resolution,
                                  "palette": "Spectral",
                                  "legend": "full",
                                  "linewidth": 2.5,
                                  "total time": total_time_across_resolution,  
                                  "csv time": csv_time_across_resolution,
                                  "json time": json_time_across_resolution,
                                  "training time": training_time_across_resolution
            }
            
            return the_dict_to_return
            
        
                            
    if doplot is True:
        
        ############################### @@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        if method == "Logistic":
            method = "Logistic Regression"
        elif method == "KNNSingle":
            method = "KNN"
        elif method == "ColdForestSingle":
            method = "Random Forest"
                
        plot.set_title(f"{state_string} state; {method}", size=18)
        ############################### @@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        return (figure, plot)


def make_resolution_line_plot(method_list, state_string_list, metric_list, reverse, doplot, path_empty_method, 
moiety_list=None, data_handling_list=[None], resolution_range=None):
    
    global data_handling_subplot_list
    sns.set_theme(style="whitegrid")
    
    figure_plot_list = []
    figure_plot_method_state_pairs_list = []
    if data_handling_list != [None]:
        data_handling_subplot_list = []
    
    for state_string in state_string_list:
        for method in method_list:
            for data_handling in  data_handling_list:
                # !!! Remember to reset the dataframe
                (moiety_dict, sought_files_df) = load_the_data(method, path_empty_method, data_handling, resolution_range)
                the_df = make_dataframe(moiety_dict, sought_files_df)

                # Let's change the dataframe a lil' bit:
                the_df.reset_index(inplace=True)
                the_df.drop(columns=["index"], inplace=True)
                
                ################################################################# @@@
                #Excluding unwanted moieties so that it's more clear
                if moiety_list is not None:
                    columns_to_retain = []
                    for column in the_df.columns:
                        ####################################################
                        # In case the moiety_list is provided:
                        moiety_name = column.split("_")[0]
                        
                        # First case are columns like resolution or state
                        if moiety_name == column:
                            columns_to_retain.append(column)
                            
                        else:
                            if moiety_name == "Carbonyl group. Low specificity":
                                moiety_name = "Carbonyl (any)"
                                
                            elif moiety_name == "Primary or secondary amine":
                                moiety_name = "Amine (any)"

                            elif moiety_name == "Non-aromatic Halide":
                                moiety_name = "Non-aromatic Halogen"

                            elif moiety_name == "Aromatic halogen":
                                moiety_name = "Aromatic Halogen"
  
                            elif moiety_name == "Aromatic nitrogen":
                                moiety_name = "Aromatic Nitrogen"

                            elif moiety_name == "Aromatic ether":
                                moiety_name = "Aromatic Ether"
                                

                            if moiety_name in moiety_list:
                                columns_to_retain.append(column)
                            else:
                                continue
                        ####################################################
                
                    the_df = the_df[columns_to_retain]
                ################################################################# @@@

                # Let's make the plot
                if doplot is True:
                    (figure, plot) = make_resolution_line_plot_data_making(the_df, state_string, method, metric_list, reverse, doplot, resolution_range)
                    figure_plot_list.append((figure, plot))
                else: 
                    data_dictionary = make_resolution_line_plot_data_making(the_df, state_string, method, metric_list, reverse, doplot, resolution_range)
                    figure_plot_list.append(data_dictionary)
                    
                method_state = (method, state_string)
                figure_plot_method_state_pairs_list.append(method_state)
                if data_handling_list != [None]:
                    data_handling_subplot_list.append(data_handling)

    #######################################################################################################################################################
    #######################################################################################################################################################
    # If doplot is False == we want a total plot made of subplots, the followinging needs to be done:
    if doplot is False:
    
        ##################################################
        # Plotting
        ##################################################
            
         # Before doing anything:
        output = get_subplot_dimensions(figure_plot_list)
        
        dimensions = output[0]
        highest_num = output[1]
        dimensions = (3,1)


        figure, plot = plt.subplots(dimensions[0], dimensions[1], figsize=(dimensions[1]*22, dimensions[0]*14))
        figure.subplots_adjust(hspace=0.7, wspace=0.15)

        grid_place_1 = -1
        grid_place_2 = 0
        
        for i in range(highest_num):
    
            grid_place_1 += 1
            
            if grid_place_1 >= dimensions[0]:
                grid_place_1 = 0
                grid_place_2 += 1
                
            #Getting data:
            subplot_data = figure_plot_list[i]
            #print(subplot_data)
            
            x_data = subplot_data["x"]
            y_data = subplot_data["y"]
            hue_data = subplot_data["hue"]
            x_ticks_data = subplot_data["x_ticks"]
            x_ticks_labels_data = subplot_data["x_ticks_labels"]
            palette_data = subplot_data["palette"]
            legend_data = subplot_data["legend"]
            linewidth_data = subplot_data["linewidth"]
                
            ############################ Finally beginning to plot ############################
            
            try:
                subplot = sns.lineplot(ax=plot[grid_place_1, grid_place_2], x=x_data, y=y_data, hue=hue_data, 
                             palette=palette_data, legend=legend_data, linewidth=linewidth_data*2)
            except:
                subplot = sns.lineplot(ax=plot[grid_place_1], x=x_data, y=y_data, hue=hue_data, 
                             palette=palette_data, legend=legend_data, linewidth=linewidth_data*2)
            
            subplot.legend(loc="upper right", fontsize=34, markerscale=3)
            subplot.set_xticks(x_ticks_data)
            subplot.set_xticklabels(x_ticks_labels_data, rotation=45, size=24)
            subplot.set_xlabel("organic moiety", size=34)
            
            # I need to get the method and the state_string, so I added the 
            # figure_plot_method_state_pairs_list is done, hurray!
            method_state = figure_plot_method_state_pairs_list[i]
            method = method_state[0]
            state_string = method_state[1]
            
            # So that naming is somewhat consistent
            if method == "Logistic":
                method = "Logistic Regression"
            elif method == "KNNSingle":
                method = "KNN"
            elif method == "ColdForestSingle":
                method = "Random Forest"
            elif method == "ColdMLP":
                method = "MLP"
                
            # The title:
            if data_handling_list != [None]:
                data_handling = data_handling_subplot_list[i]
                subplot.set_title(f"{state_string} state; {method}; {data_handling}", size=44)
            else:
                subplot.set_title(f"{state_string} state; {method}", size=44)
                
            ############################ MAKING IT PRETTIER ############################
            
            # The lower value is the 20% of the difference between the upper value - always 1.0 and the minimum of y_axis
            y_minimum = min(y_data)
            lower_limit = y_minimum - 0.2*(1-y_minimum)
            if lower_limit < 0:
                lower_limit = 0
                
            subplot.set_ylim(lower_limit, 1.0)
            subplot.set_xlim(x_data[0], x_data[-1])
            # Yeah, unfortunately I need to manually set the y_ticks
            y_ticks = np.linspace(lower_limit+0.1*(1-y_minimum), 1.0-0.1*(1-y_minimum), 8)
            ###############################
            # VERY, VERY IMPORTANT. This format things ensures that the label numbers INCLUDE trailing zeros
            y_ticks_labels = ['{:.2f}'.format(num) for num in np.around(y_ticks, decimals=2)]
            subplot.set_yticks(y_ticks)
            subplot.set_yticklabels(y_ticks_labels, size=24)
            
            subplot.set_ylabel("average precision score value", size=34)
            
            # plot[grid_place_1, grid_place_2].legend(loc='center right', bbox_to_anchor=(1.15, 0.5), fontsize=18)
            
            ###################################################################################################################
            ###################################################################################################################
           
    else:
        pass

    return None


def make_time_metric_scatter_plot(method_list, state_string_list, metric_list, reverse, path_empty_method,
                                  x_normalized, y_normalized, make_mean, Just_Give_Data, 
                                  moiety_list=None, data_handling_list=[None], resolution_range=None):
    
    global data_handling_subplot_list
    sns.set_theme(style="whitegrid")
    
    figure_plot_list = []
    figure_plot_method_state_pairs_list = []
    if data_handling_list != [None]:
        data_handling_subplot_list = []
    
    for state_string in state_string_list:
        for method in method_list:
            for data_handling in  data_handling_list:

                # Here is, of course, the super important part of making the dataframe. This also the moment that I realised that this is where
                # the changes have to be made in order to also obtain the data values of training time
                # I decided that the easiest way is to simply add three columns to the sought_files_dataframe
       
                (moiety_dict, sought_files_df) = load_the_data(method, path_empty_method, data_handling, resolution_range)
                the_df = make_dataframe(moiety_dict, sought_files_df)

                # Let's change the dataframe a lil' bit:
                the_df.reset_index(inplace=True)
                the_df.drop(columns=["index"], inplace=True)
                
                ################################################################# @@@
                #Excluding unwanted moieties so that it's more clear
                if moiety_list is not None:
                    columns_to_retain = []
                    for column in the_df.columns:
                        ####################################################
                        # In case the moiety_list is provided:
                        moiety_name = column.split("_")[0]
                        
                        # First case are columns like resolution or state
                        if moiety_name == column:
                            columns_to_retain.append(column)
                            
                        else:
                            if moiety_name in moiety_list:
                                columns_to_retain.append(column)
                            else:
                                continue
                        ####################################################
                
                    the_df = the_df[columns_to_retain]
                ################################################################# @@@
                
                # Let's get the data dictionary, from which we will get the data to make the plot

                doplot=False
                data_dictionary = make_resolution_line_plot_data_making(the_df, state_string, method, metric_list, reverse, doplot, resolution_range)
                figure_plot_list.append(data_dictionary)
                    
                method_state = (method, state_string)
                figure_plot_method_state_pairs_list.append(method_state)
                if data_handling_list != [None]:
                    data_handling_subplot_list.append(data_handling)
                    
                    
                                
    ##################################################
    #                  Plotting                      #
    ##################################################

        
    # Before doing anything, let's get the dimensions:
    output = get_subplot_dimensions(figure_plot_list)
    
    dimensions = output[0]
    dimensions=(3,1)
    highest_num = output[1]

    figure, plot = plt.subplots(dimensions[0], dimensions[1], figsize=(dimensions[1]*22, dimensions[0]*14))
    figure.subplots_adjust(hspace=0.4, wspace=0.15)

    grid_place_1 = -1
    grid_place_2 = 0
    
    for i in range(highest_num):

        grid_place_1 += 1
        
        if grid_place_1 >= dimensions[0]:
            grid_place_1 = 0
            grid_place_2 += 1

        ############################## @@@
        # Getting data:
        # The original make_resolution_line_plot_data_making function returns the following values in a dictionary:
        # x --> Organic moiety
        # y --> Metric value
        # hue --> the resolution
        # title (singular subplot) --> physical state, ML method, data_handling

        # The time scatter plot has different axes:
        # x --> Time
        # y --> Metric value
        # hue --> The resolution
        # title (singular subplot) --> Same as above (should be combined)

        # The only difference between the two is the x-axis.
        # The make_resolution_line_plot_data_making function must therefore convey the values of time.
        ############################## @@@


        subplot_data = figure_plot_list[i]
        # The crux:
        #########
        x_data = subplot_data["total time"]
        #########
        
        y_data = subplot_data["y"]
        hue_data = subplot_data["hue"]
        
        
        if make_mean is True:
            df_for_mean = pd.DataFrame(data={"x_data": x_data, "y_data": y_data, "hue_data": hue_data})
            the_groups = df_for_mean.groupby("hue_data")
            
            # Creating the new lists
            x_data = []
            y_data = []
            hue_data = []
        
            for group, df_for_mean_grouped in the_groups:
                # The x_data contains the same values, hue value too, obviously
                # So really, the only data that will have differenct values is y_data as expected
                # But for the sake of clarity let's get the mean for everything, even if getting the mean for x_data and hue_data is redundant
                
                x_data.append(np.mean(df_for_mean_grouped["x_data"]))
                y_data.append(np.mean(df_for_mean_grouped["y_data"]))
                hue_data.append(np.mean(df_for_mean_grouped["hue_data"]))
        
        if Just_Give_Data is True:
            
            plt.close()
            return x_data, y_data, hue_data
        
        # Let's not forget about figure_plot_method_state_pairs_list and the data_handling_subplot_list
   
        ############################ Finally beginning to plot ############################
        try:
            subplot = sns.scatterplot(ax=plot[grid_place_1, grid_place_2], x=x_data, y=y_data, hue=hue_data, legend=hue_data, s=2000)
        except:
            subplot = sns.scatterplot(ax=plot[grid_place_1], x=x_data, y=y_data, hue=hue_data, legend=hue_data, s=2000)
        
        
        ################# The title:
        method_state = figure_plot_method_state_pairs_list[i]
        method = method_state[0]
        state_string = method_state[1]
        
        # So that naming is somewhat consistent
        if method == "Logistic":
            method = "Logistic Regression"
        elif method == "KNNSingle":
            method = "KNN"
        elif method == "ColdForestSingle":
            method = "Random Forest"
        elif method == "ColdMLP":
            method = "MLP"
            
        # The title:
        if data_handling_list != [None]:
            data_handling = data_handling_subplot_list[i]
            subplot.set_title(f"{state_string} state; {method}; {data_handling}", size=44)
        else:
            subplot.set_title(f"{state_string} state; {method}", size=44)
            
            
        ################# The legend:
        subplot.legend(loc="upper right", bbox_to_anchor=(1.05, 0.9), fontsize=34, markerscale=3)
        
        ################# The normalisation:
        if x_normalized is True:

            ################################################ NORMALIZATION
            scaler = MinMaxScaler(feature_range=(0.1, 0.9)).fit(np.array(x_data).reshape(-1, 1))
            x_data = scaler.transform(np.array(x_data).reshape(-1, 1))

            # I know it looks weird, but that is the way. The x[0,:] looks like this:
            # [0.32473623]
            # [0.2]
            # [0.29411765]
            # [0.25548246]

            x_list = []
            for i in range(len(x_data)):
                x_list.append(float(x_data[i, 0]))

            x_data = x_list
            ################################################ NORMALIZATION

        if y_normalized is True:

            ################################################ NORMALIZATION
            scaler = MinMaxScaler(feature_range=(0.05, 0.95)).fit(np.array(y_data).reshape(-1, 1))
            y_data = scaler.transform(np.array(y_data).reshape(-1, 1))
            y_list = []
            for i in range(len(y_data)):
                y_list.append(float(y_data[i, 0]))

            y = y_list
            ################################################ NORMALIZATION
      
      
      
        ###################################################################################################################
        ########################################## MAKING IT PRETTIER #####################################################
        #The lower value is the 20% of the difference between the upper value - maximum of y_axis and the minimum of y_axis
        y_minimum = min(y_data)
        y_maximum = max(y_data)
        lower_limit = y_minimum - 0.2*(y_maximum-y_minimum)
        if lower_limit < 0:
            lower_limit = 0
        upper_limit = y_maximum + 0.2*(y_maximum-y_minimum)
        if upper_limit > 1:
            upper_limit = 1
        subplot.set_ylim(lower_limit, upper_limit)
        # Yeah, unfortunately I need to manually set the y_ticks
        if y_maximum-y_minimum > 0.1:
            y_ticks = np.linspace(lower_limit+0.1*(y_maximum-y_minimum), upper_limit-0.1*(y_maximum-y_minimum), 8)
        else:
            y_ticks = np.linspace(lower_limit+0.05*(y_maximum-y_minimum), upper_limit-0.05*(y_maximum-y_minimum), 8)
        ###############################
        # VERY, VERY IMPORTANT. This format things ensures that the label numbers INCLUDE trailing zeros
        y_ticks_labels = ['{:.4f}'.format(num) for num in np.around(y_ticks, decimals=4)]
        subplot.set_yticks(y_ticks)
        subplot.set_yticklabels(y_ticks_labels, size=28)
        
        
        #I will do the same for the x_axis
        x_minimum = min(x_data)
        x_maximum = max(x_data)
        lower_limit = x_minimum - 0.2*(x_maximum-x_minimum)
        if lower_limit < 0:
            lower_limit = 0
        upper_limit = x_maximum + 0.2*(x_maximum-x_minimum)
        subplot.set_xlim(lower_limit, upper_limit)
        # Yeah, unfortunately I need to manually set the y_ticks
        if x_maximum-x_minimum > 0.1:
            x_ticks = np.linspace(lower_limit+0.1*(x_maximum-x_minimum), upper_limit-0.1*(x_maximum-x_minimum), 8)
        else:
            x_ticks = np.linspace(lower_limit+0.05*(x_maximum-x_minimum), upper_limit-0.05*(x_maximum-x_minimum), 8)
        ###############################
        # VERY, VERY IMPORTANT. This format things ensures that the label numbers INCLUDE trailing zeros
        x_ticks_labels = ['{:.2f}'.format(num) for num in np.around(x_ticks, decimals=2)]
        subplot.set_xticks(x_ticks)
        subplot.set_xticklabels(x_ticks_labels, size=28)
        
        
        subplot.set_xlabel("total execution time (s)", size=34)
        subplot.set_ylabel("average precision mean score value", size=34)
        ###################################################################################################################
        ###################################################################################################################
        
        

    return None


     
    
    
    
    
    
    
    
    
############################# BIN BIN BIN BIN ##################################

#  ######################################################################################## @@@
#    # Here the file of interest is determined:
#    sought_string_file = f"Summary600_3800_(.+)_(.+)_True_False_False_False_{method}(\d).txt"
#    ######################################################################################## @@@
#
#    #######################################################################
#    # Here the data frame is created that stores the information about the resolution, state, number, and the file string:
#    # At the time only 2.0 resolution available
#    # Numbers are from 1 to 5
#    sought_files_dict = {"resolution":[], "state":[], "number":[], "string":[]}
#    for file_string in all_files_list:
#        the_find = re.findall(sought_string_file, file_string)
#        if len(the_find) > 0:
#            resolution = the_find[0][0]
#            state = the_find[0][1]
#            number = the_find[0][2]
#
#            sought_files_dict["resolution"].append(resolution)
#            sought_files_dict["state"].append(state)
#            sought_files_dict["number"].append(number)
#            ######################################################################################## @@@
#            the_file_string_to_append = f"Summary600_3800_{resolution}_{state}_True_False_False_False_{method}{number}.txt"
#            sought_files_dict["string"].append(the_file_string_to_append)
#            ######################################################################################## @@@                               
#
#    sought_files_df = pd.DataFrame(data=sought_files_dict)
#    #######################################################################
