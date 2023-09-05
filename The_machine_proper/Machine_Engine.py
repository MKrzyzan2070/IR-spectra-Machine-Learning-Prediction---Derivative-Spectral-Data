import numpy as np
import random
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from The_machine_proper import Writing_Summary
from sklearn.decomposition import PCA
import pandas as pd

def Tackle_Class_Imbalance(X, Y, functional_group, div):
    try:
        Y = pd.DataFrame(data=Y, columns=[functional_group])
    except:
        pass
    zeros = Y[Y[functional_group] == 0]
    ones = Y[Y[functional_group] == 1]
    if len(ones)/len(zeros) < 1/div:
        index_list = list(ones.index)
        how_many_index_to_add = div * len(ones)
        zero_index_list = list(zeros.index)
        for i in range(how_many_index_to_add):
            random_num = random.randint(0, len(zero_index_list) - 1)
            index_to_add = zero_index_list[random_num]
            index_list.append(index_to_add)
            zero_index_list.remove(index_to_add)

        Y = pd.DataFrame(Y).iloc[index_list]
        X = pd.DataFrame(X).iloc[index_list]

    elif len(zeros)/len(ones) < 1/div:
        index_list = list(zeros.index)
        how_many_index_to_add = div * len(zeros)
        one_index_list = list(ones.index)
        for i in range(how_many_index_to_add):
            random_num = random.randint(0, len(one_index_list) - 1)
            index_to_add = one_index_list[random_num]
            index_list.append(index_to_add)
            one_index_list.remove(index_to_add)
        Y = pd.DataFrame(Y).iloc[index_list]
        X = pd.DataFrame(X).iloc[index_list]

    zeros = Y[Y[functional_group] == 0]
    ones = Y[Y[functional_group] == 1]
    if len(zeros)/len(ones) < 1:
        imbalance_ratio = len(zeros)/len(ones)
    elif len(ones)/len(zeros) <= 1:
        imbalance_ratio = len(ones)/len(zeros)
    else:
        imbalance_ratio = 1

    return X, Y, imbalance_ratio

def Engine_Core(functional_group, Y, X, Scoring,
                SVM, Logistic, ForestSingle, ColdMLP, KNNSingle, GaussianNB, GradientBoost,
                Class_weight, SMOTE):

    #Only because I don't want to get warnings:
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    # The data needs to be transformed from its current format into a proper dataframe for further processing and analysis.
    Y = Y.to_frame(name=functional_group)
    if len(Y[Y[functional_group] == 1]) < 10:
        print("Too few molecules")
        return None, None, None, None, None, None

    # The div is here stated. 1/9 means that the minority account for 10%
    ##################################################################################### @@@
    div = 9
    ##################################################################################### @@@

    if SMOTE is False:

        imbalance_output = Tackle_Class_Imbalance(X, Y, functional_group, div)
        X_output = imbalance_output[0]
        Y_output = imbalance_output[1]
        imbalance_ratio = imbalance_output[2]
        zeros = Y_output[Y_output[functional_group] == 0]
        ones = Y_output[Y_output[functional_group] == 1]

        #This thing will be needed later on
        more_zeros = False
        more_ones = False
        do_nothing = False

        if len(ones)/(len(zeros)) < 1/2:
            more_zeros = True
        elif len(zeros)/len(ones) <= 1/2:
            more_ones = True
        else:
            do_nothing = True

        #################################################### @@@
        #################################################### @@@
        # Change it, if you wish to see the feature importance:
        if ForestSingle is False or ColdMLP is False:

            print("Performing the PCA")
            PCA_components = min([int(len(X_output.columns) * 0.05), int(len(X_output) * 0.05)])
            if PCA_components > 5:
                pca = PCA(n_components=PCA_components)
            else:
                pca = PCA(n_components=5)

            print("Fitting")
            pca.fit(X_output)
            print("Transforming")
            X_output = pca.transform(X_output)
            print(f"PCA components number: {pca.n_components_}")
            del pca

        #################################################### @@@
        #################################################### @@@

        try:
            x_train, x_test, y_train, y_test = train_test_split(X_output, Y_output.values.ravel(), train_size=0.75, stratify=Y_output.values.ravel())
        except:
            x_train, x_test, y_train, y_test = train_test_split(X_output, Y_output.ravel(), train_size=0.75, stratify=Y_output.ravel())

        ########################################################################
        if Class_weight is True:
            class_weight_dict = {}

            if do_nothing is False:
                if more_zeros is True:
                    class_weight_dict[0] = 1
                    class_weight_dict[1] = round(1 / imbalance_ratio, 2)
                elif more_ones is True:
                    class_weight_dict[0] = round(1 / imbalance_ratio, 2)
                    class_weight_dict[1] = 1
            else:
                class_weight_dict[0] = 1
                class_weight_dict[1] = 1
        else:
            #Of course if the class_weight is not demanded it's value is None
            class_weight_dict = None
        ########################################################################


    ########################################################################
    else:
        ##################################
        ##################################
        #### HERE COMES THE SMOTE ########
        ##################################
        ##################################

        from imblearn.over_sampling import SMOTE
        smote = SMOTE(sampling_strategy='minority')

        # To ensure that the TackleClassImbalance method is comparable with the SMOTE method, the test data
        # should have the same ratio. Therefore, it's necessary to split the train and test data beforehand.

        #################################################### @@@
        #################################################### @@@
        # Change it, if you wish to see the feature importance:
        if ForestSingle is False or ColdMLP is False:
            print("Performing the PCA")
            PCA_components = min([int(len(X.columns) * 0.05), int(len(X) * 0.05)])
            if PCA_components > 5:
                pca = PCA(n_components=PCA_components)
            else:
                pca = PCA(n_components=5)

            print("Fitting")
            pca.fit(X)
            print("Transforming")
            X = pca.transform(X)
            print(f"PCA components number: {pca.n_components_}")
            del pca
        #################################################### @@@
        #################################################### @@@

        try:
            x_train, x_test, y_train, y_test = train_test_split(X, Y.values.ravel(), train_size=0.75, stratify=Y.values.ravel())
        except:
            x_train, x_test, y_train, y_test = train_test_split(X, Y.ravel(), train_size=0.75, stratify=Y.ravel())

        ########################################################################
        if Class_weight is True:
            print("When selecting SMOTE, having Class weights is NOT POSSIBLE !!!")
            class_weight_dict = None
        else:
            # Of course if the class_weight is not demanded it's value is None
            class_weight_dict = None
        ########################################################################

        X_output_train, Y_output_train = smote.fit_resample(x_train, y_train)

        #Redifining:
        x_train = X_output_train
        y_train = Y_output_train

        ##################
        #Ensuring the correct ratio. But it just is for x_test data!!!
        ###################
        imbalance_output = Tackle_Class_Imbalance(x_test, y_test, functional_group, div)
        X_output_test = imbalance_output[0]
        Y_output_test = imbalance_output[1]
        imbalance_ratio = imbalance_output[2]

        #Redifining:
        x_test = X_output_test
        y_test = Y_output_test

        #There needs to be X_output for it to work so:
        X_output = X
        Y_output = Y
    ########################################################################



    #################################################################################
    ######################### THE REAL CORE OF THE MACHINE ##########################
    #################################################################################

    ##################
    # The Logistic regression:
    if Logistic is True:
        from sklearn.linear_model import LogisticRegression

        #Sometimes has convergence errors
        
        #################### @@@
        grid_values = {"C": [3]}
        #################### @@@
        if class_weight_dict is None:
            class_weight_dict = "balanced"
            
        classifier = LogisticRegression(solver="saga", class_weight=class_weight_dict, max_iter=100)
        
        #First determining the best parameter values:
        grid_classifier = GridSearchCV(classifier, param_grid=grid_values, scoring=Scoring).fit(x_train, y_train)
        best_parameters = grid_classifier.best_params_
        best_score = grid_classifier.best_score_
        
        #Now stating the proper parameter values:
        grid_classifier = LogisticRegression(solver="saga", C=best_parameters["C"], class_weight=class_weight_dict)


    ##################
    # The SVM:
    elif SVM is True:
        from sklearn.svm import SVC


        #################### @@@
        grid_values = {"C": [1],
                       "gamma": ["scale"]}
        #################### @@@
        if class_weight_dict is None:
            class_weight_dict = "balanced"
        
        classifier = SVC(kernel="rbf", class_weight=class_weight_dict)
        
        #First determining the best parameter values:
        grid_classifier = GridSearchCV(classifier, param_grid=grid_values, scoring=Scoring).fit(x_train, y_train)
        best_parameters = grid_classifier.best_params_
        best_score = grid_classifier.best_score_
        
        #Now stating the proper parameter values:
        grid_classifier = SVC(probability=True, kernel="rbf", C=best_parameters["C"], gamma=best_parameters["gamma"], class_weight=class_weight_dict)


    elif GradientBoost is True:
        from sklearn.ensemble import HistGradientBoostingClassifier

        classifier = HistGradientBoostingClassifier()
        grid_classifier = classifier.fit(x_train, y_train)
        best_parameters = grid_classifier.best_params_
        best_score = grid_classifier.best_score_
        grid_classifier = HistGradientBoostingClassifier(learning_rate=best_parameters["learning_rate"], max_depth=best_parameters["max_depth"])


    ##################
    # The KNN:
    elif KNNSingle is True:
        from sklearn.neighbors import KNeighborsClassifier

        #################### @@@
        grid_values = {"n_neighbors": [3]}
        #################### @@@
        if class_weight_dict is None:
            class_weight_dict = "distance"
            
        classifier = KNeighborsClassifier(weights=class_weight_dict)
        
        #First determining the best parameter values:
        grid_classifier = GridSearchCV(classifier, param_grid=grid_values, scoring=Scoring).fit(x_train, y_train)
        best_parameters = grid_classifier.best_params_
        best_score = grid_classifier.best_score_
        
        #Now stating the proper parameter values:
        grid_classifier = KNeighborsClassifier(weights=class_weight_dict, n_neighbors=best_parameters["n_neighbors"])


    elif GaussianNB is True:
        from sklearn.naive_bayes import GaussianNB

        classifier = GaussianNB()
        grid_classifier = classifier.fit(x_train, y_train)
        best_parameters = None
        best_score = None
        

    ##################
    # The forest:
    elif ForestSingle is True:
        from sklearn.ensemble import RandomForestClassifier

        #################### @@@
        grid_values = {"max_depth": [10],
                       "n_estimators": [100],
                       "max_features": ["sqrt"],
                       "min_samples_split": [5],
                       "min_samples_leaf": [3]
                        
        }
        #################### @@@
        if class_weight_dict is None:
            class_weight_dict = "balanced"
            
        classifier = RandomForestClassifier(class_weight=class_weight_dict)

        #First determining the best parameter values:
        grid_classifier = GridSearchCV(classifier, param_grid=grid_values, scoring=Scoring).fit(x_train, y_train)
        best_parameters = grid_classifier.best_params_
        best_score = grid_classifier.best_score_
        
        #Now stating the proper parameter values:
        grid_classifier = RandomForestClassifier(max_depth=best_parameters["max_depth"], n_estimators=best_parameters["n_estimators"], 
        max_features=best_parameters["max_features"], min_samples_split=best_parameters["min_samples_split"], min_samples_leaf=best_parameters["min_samples_leaf"]
        ,class_weight=class_weight_dict)

    # The MLP:
    elif ColdMLP is True:
        from sklearn.neural_network import MLPClassifier
        
        # For large, complicated datasets with a lot of features, it is recommended to use 
        # stochastic gradient descent (SGD) solver 
        # with the "log" or "modified_huber" loss function. 
        
        #################### @@@
        grid_values = { "solver": ["lbfgs"],
                        "activation": ["relu"],
                        "alpha": [10],
                        "hidden_layer_sizes": [(100, 50)]
        }
        #################### @@@

        # You can't state the class_weights in the MLP, but let's just write it
        # regardless for the sake of consistency
        if class_weight_dict is None:
            class_weight_dict = "balanced"
        
        classifier = MLPClassifier(learning_rate="adaptive", max_iter=800).fit(x_train, y_train)
        
        #First determining the best parameter values:
        grid_classifier = GridSearchCV(classifier, param_grid=grid_values, scoring=Scoring).fit(x_train, y_train)
        best_parameters = grid_classifier.best_params_
        best_score = grid_classifier.best_score_
        
        #Now stating the proper parameter values:
        grid_classifier = MLPClassifier(learning_rate="adaptive", max_iter=800, solver=best_parameters["solver"], activation=best_parameters["activation"], 
        alpha=best_parameters["alpha"], hidden_layer_sizes=best_parameters["hidden_layer_sizes"])


    return x_train, x_test, y_train, y_test, grid_classifier, Y, X_output, Y_output, best_score, best_parameters, SMOTE, imbalance_ratio


def Warm_Start_Engine_Core(functional_group, Y, X, Perceptron, WarmForestSingle, MLP, SMOTE):

    # Because it's converted to this weird thing but needs to be a proper dataframe
    Y = Y.to_frame(name=functional_group)
    if len(Y[Y[functional_group] == 1]) < 10:
        print("Too few molecules")
        return None, None, None, None, None, None

    # In other cases to tackle imbalance this needs to be applied:
    ##################################################################################### @@@
    div = 7
    ##################################################################################### @@@

    output = Tackle_Class_Imbalance(X, Y, functional_group, div)
    X_output = output[0]
    Y_output = output[1]

    # Doing it first time:
    x_train, x_test, y_train, y_test = train_test_split(X_output, Y_output.values.ravel(), train_size=0.8, stratify=Y_output.values.ravel())

    div = 1
    output = Tackle_Class_Imbalance(x_train, y_train, functional_group, div)
    X_output_warm = output[0]
    Y_output_warm = output[1].values.ravel()
    imbalance_ratio = output[2]

    how_many_times = int(5/imbalance_ratio)

    best_parameters = None
    best_score = None

    if Perceptron is True:
        from sklearn.linear_model import Perceptron
        classifier = Perceptron(n_iter_no_change=20 ,warm_start=True)
        classifier.fit(X_output_warm, Y_output_warm)

        if how_many_times > 1:
            for i in range(how_many_times):

                output = Tackle_Class_Imbalance(x_train, y_train, functional_group, div)
                X_output_warm = output[0]
                Y_output_warm = output[1].values.ravel()

                classifier.fit(X_output_warm, Y_output_warm)


    elif WarmForestSingle is True:
        from sklearn.ensemble import RandomForestClassifier
        #The Grid_values need to be ditched. Sorry
        if how_many_times < 1:
            number_est = 200
        else:
            number_est = 10
        classifier = RandomForestClassifier(n_estimators=number_est, warm_start=True, max_depth=10,
                                            min_samples_split=2, min_samples_leaf=2)
        classifier.fit(X_output_warm, Y_output_warm)


        # Building on it: I think 0.5 is a reasonable figure
        if how_many_times > 1:
            for i in range(how_many_times):

                output = Tackle_Class_Imbalance(x_train, y_train, functional_group, div)
                X_output_warm = output[0]
                Y_output_warm = output[1].values.ravel()

                classifier.n_estimators += 10
                classifier.fit(X_output_warm, Y_output_warm)

    elif MLP is True:
        from sklearn.neural_network import MLPClassifier

        classifier = MLPClassifier(warm_start=True, max_iter=5000).fit(x_train, y_train)
        if how_many_times > 1:
            for i in range(how_many_times):
                output = Tackle_Class_Imbalance(x_train, y_train, functional_group, div)
                X_output_warm = output[0]
                Y_output_warm = output[1].values.ravel()

                classifier.fit(X_output_warm, Y_output_warm)

    # Changes to measurement:
    return x_train, x_test, y_train, y_test, classifier, Y, X_output, Y_output, best_score, best_parameters, SMOTE, imbalance_ratio



def Snip_it(X, X_columns, functional_group):
    import re
    import json
    # Ok so we have the df and the columns:
    # The columns are potentially: Normal, D:, Cat-, Cat-D:
    with open("The_machine_proper/FunctionalGroupsWeights.json", 'r') as openfile:
        weights_dict = json.load(openfile)

    if isinstance(functional_group, list):
        #If it is a list, then no problem, simply making one giga-list
        add_weight_from_list = []
        add_weight_to_list = []
        for func_group in functional_group:
            add_weight_from_list_sub = weights_dict[func_group]["from"]
            add_weight_to_list_sub = weights_dict[func_group]["to"]

            for add_weight_from in add_weight_from_list_sub:
                add_weight_from_list.append(add_weight_from)
            for add_weight_to in add_weight_to_list_sub:
                add_weight_to_list.append(add_weight_to)

    else:
        add_weight_from_list = weights_dict[functional_group]["from"]
        add_weight_to_list = weights_dict[functional_group]["to"]

    if len(add_weight_from_list) > 0 and len(add_weight_to_list) > 0:

        new_X_columns = []
        for column in X_columns:
            # Let's simply look for the 4-digit numbers and that's all
            the_find = re.findall("\d{4}", column)
            out_of_boundaries = True

            # Checks if it's in any of the boundaries:
            # If any number falls within any of the boundaries, then it's ok
            for i in range(len(add_weight_from_list)):
                add_weight_from = add_weight_from_list[i]
                add_weight_to = add_weight_to_list[i]
                for number in the_find:
                    number = int(number)
                    if number >= add_weight_from and number <= add_weight_to:
                        out_of_boundaries = False

            if out_of_boundaries is False:
                new_X_columns.append(column)
            else:
                continue

        for old_column in X_columns:
            if old_column not in new_X_columns:
                X.drop(columns=old_column, inplace=True)

        print(X.columns)
        return X

    else:
        return X

def Make_Dummy_List(x_train, y_train):
    from sklearn.dummy import DummyClassifier

    strategy_list = ["most_frequent", "stratified", "uniform"]
    dummy_list = []
    for strategy in strategy_list:
        dummy_clf = DummyClassifier(strategy=strategy).fit(x_train, y_train)
        dummy_list.append(dummy_clf)

    return dummy_list

def oneD_array_engine(df, Y_columns, X_columns, Snipping, Scoring,
                              SVM, Logistic, ColdForestSingle, ColdMLP, KNNSingle, GaussianNB, GradientBoost,
                      WarmForestSingle, Perceptron, WarmMLP, Class_Weight, SMOTE):

    for num in range(len(Y_columns)):
        X = df[X_columns]
        functional_group = Y_columns[num]
        Y = df[functional_group]

        print(f"The {functional_group} functional group begins")
        # Now let's add the weights for the functional groups:
        if Snipping is True:
            X = Snip_it(X, X_columns, functional_group)

        #####################################################################################################
        ########################   THE CORE:  ###############################################################
        if Perceptron is True or WarmForestSingle is True or WarmMLP is True:
            (x_train, x_test, y_train, y_test, grid_classifier, Y, X_output, Y_output, best_Score, best_parameters, SMOTE) = Warm_Start_Engine_Core(functional_group, Y, X, Perceptron,
                                                                                            WarmForestSingle, WarmMLP, SMOTE)
            #grid_classifier = Warm_Start_Engine_Core(functional_group, Y, X, Perceptron, WarmForestSingle)

        else:
            (x_train, x_test, y_train, y_test, grid_classifier, Y, X_output, Y_output, best_Score, best_parameters, SMOTE, imbalance_ratio) = Engine_Core(functional_group, Y, X, Scoring,
                                                                              SVM, Logistic, ColdForestSingle, ColdMLP, KNNSingle, GaussianNB,
                                                                              GradientBoost, Class_Weight, SMOTE)
        #####################################################################################################
        #####################################################################################################

        # Let's do the dummy score:
        dummy_list = Make_Dummy_List(x_train, y_train)

        DoVersus = False
        Writing_Summary.Write_Summary(DoVersus, num, functional_group, grid_classifier, x_train, y_train,
                                      x_test, y_test, Y, X_output, Y_output, dummy_list, Scoring, best_Score, best_parameters, SMOTE, imbalance_ratio)
        print("Ended...")



def oneD_array_engine_Mix(MixOnly, OneD_Mixing_No_Other_Group, df, Y_columns, Y_Mixlist, X_columns, Snipping, Scoring,
                                      SVM, Logistic, ColdForestSingle, ColdMLP,KNNSingle, GaussianNB,
                          GradientBoost, WarmForestSingle, Perceptron, WarmMLP, Class_Weight, SMOTE):

    num = -1
    for mix_group in Y_Mixlist:
        num += 1
        X = df[X_columns]
        print(f"The {mix_group} mixed functional group begins")
        # Now let's add the weights for the functional groups:
        functional_group_mix = ""
        functional_group_mix_list = []
        for group in mix_group:
            functional_group_mix = functional_group_mix + " " + group
            functional_group_mix_list.append(group)

        if OneD_Mixing_No_Other_Group is False:
            def fun_to_apply(dataframe):
                presence_list = []
                for group in mix_group:
                    presence_list.append(dataframe[group])
                if len(presence_list) == presence_list.count(1):
                    return 1
                else:
                    return 0

            df[functional_group_mix] = df.apply(fun_to_apply, axis=1)
            Y = df.apply(fun_to_apply, axis=1)

        else:
            def fun_to_apply(dataframe):
                presence_list = []
                for group in mix_group:
                    presence_list.append(dataframe[group])
                if len(presence_list) == presence_list.count(1):
                    other_groups_presence = []
                    for func_group in Y_columns:
                        # I have to make exception for "Carbonyl group.Low specificity"
                        # And also the exception for the aromatics and alkane, since they can be always present
                        if (func_group not in mix_group) and (func_group != "Carbonyl group.Low specificity") \
                                and (func_group != "Arene") and (func_group != "Alkane"):
                            other_groups_presence.append(dataframe[func_group])
                    if other_groups_presence.count(1) == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0

            if len(mix_group) == 1 and OneD_Mixing_No_Other_Group is True:
                functional_group_mix = functional_group_mix + " clean"

            print(functional_group_mix)
            df[functional_group_mix] = df.apply(fun_to_apply, axis=1)
            Y = df.apply(fun_to_apply, axis=1)

        check_df = df[df[functional_group_mix] == 1]
        print(check_df["Name"])

        #Also Snipping must be done:
        if Snipping is True:
            Snip_it(X, X_columns, functional_group_mix_list)


        #####################################################################################################
        ########################   THE CORE:  ###############################################################
        if Perceptron is True or WarmForestSingle is True or WarmMLP is True:
            (x_train, x_test, y_train, y_test, grid_classifier, Y, X_output, Y_output, best_Score, best_parameters, SMOTE) = Warm_Start_Engine_Core(functional_group_mix, Y, X, Perceptron,
                                                                                            WarmForestSingle, WarmMLP, SMOTE)

        else:
            (x_train, x_test, y_train, y_test, grid_classifier, Y, X_output, Y_output, best_score, best_parameters, SMOTE, imbalance_ratio) = Engine_Core(functional_group_mix, Y, X, Scoring,
                                                                              SVM, Logistic, ColdForestSingle, ColdMLP, KNNSingle,
                                                                              GaussianNB,
                                                                              GradientBoost, Class_Weight, SMOTE)
        #####################################################################################################
        #####################################################################################################

        # Let's do the dummy score:
        dummy_list = Make_Dummy_List(x_train, y_train)

        # Summary time:
        DoVersus = False
        Writing_Summary.Write_Summary(DoVersus, num, functional_group_mix, grid_classifier, x_train, y_train,
                                      x_test, y_test, Y, X_output, Y_output, dummy_list, Scoring, best_score, best_parameters, SMOTE, imbalance_ratio)
        print("Ended...")


    if MixOnly is False:
        Summary_before = True
        oneD_array_engine(df, Y_columns, X_columns, Snipping, Scoring,
                              SVM, Logistic, ColdForestSingle, ColdMLP, KNNSingle, GaussianNB, GradientBoost, WarmForestSingle, Perceptron, WarmMLP, Class_Weight, SMOTE)


def Do_The_Versus(df, Versus1, Versus2, X_columns, Snipping, Scoring,
                          SVM, Logistic, ColdForestSingle, ColdMLP, KNNSingle, GaussianNB, GradientBoost,
                  WarmForestSingle, Perceptron, WarmMLP, Class_Weight, SMOTE):


    num = -1
    for i in Versus1:
        num += 1
        print(f"The {Versus1[num]} functional group vs {Versus2[num]} begins")
        # Now let's add the weights for the functional groups:
        functional_group = ""
        for group in Versus1[num]:
            functional_group = functional_group + group
        functional_group = functional_group + " vs. "
        for group in Versus2[num]:
            functional_group = functional_group + group

        def fun_to_apply(dataframe):
            counting_list = []
            for group in Versus1[num]:
                counting_list.append(dataframe[group])
            if len(counting_list) == counting_list.count(1):
                return 1

            counting_list = []
            for group in Versus2[num]:
                counting_list.append(dataframe[group])
            if len(counting_list) == counting_list.count(1):
                return 0
            else:
                return np.NAN

        Y = df.apply(fun_to_apply, axis=1)
        Y.dropna(inplace=True)
        index_list = Y.index
        X = df.loc[index_list, X_columns]

        #####################################################################################################
        ########################   THE CORE:  ###############################################################
        if Perceptron is True or WarmForestSingle is True or WarmMLP is True:
            (x_train, x_test, y_train, y_test, grid_classifier, Y, X_output, Y_output, best_score, best_parameters, SMOTE) = Warm_Start_Engine_Core(functional_group, Y, X, Perceptron,
                                                                                            WarmForestSingle, WarmMLP, SMOTE)

        else:
            (x_train, x_test, y_train, y_test, grid_classifier, Y, X_output, Y_output, best_score, best_parameters, SMOTE, imbalance_ratio) = Engine_Core(functional_group, Y, X, Scoring,
                                                                              SVM, Logistic, ColdForestSingle, ColdMLP, KNNSingle,
                                                                              GaussianNB,
                                                                              GradientBoost, Class_Weight, SMOTE)
        #####################################################################################################
        #####################################################################################################

        # Let's do the dummy score:
        dummy_list = Make_Dummy_List(x_train, y_train)

        # Summary time:
        DoVersus = True
        Writing_Summary.Write_Summary(DoVersus, num, functional_group, grid_classifier, x_train, y_train,
                                      x_test, y_test, Y, X_output, Y_output, dummy_list, Scoring, best_score, best_parameters, SMOTE, imbalance_ratio)
        print("Ended...")








def MultiLabel(df, Y_columns, X_columns, Scoring, KNNMulti):

    X = df[X_columns]
    Y = df[Y_columns]


    x_train, x_test, y_train, _ = train_test_split(X, Y, train_size=0.8)

    ####################################################################################
    ####################################################################################
    ####################################################################################

