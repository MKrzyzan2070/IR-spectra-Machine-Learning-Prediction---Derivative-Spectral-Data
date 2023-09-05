import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=Warning)


def Write_Summary(DoVersus, num, functional_group, grid_svc, x_train, y_train, x_test, y_test, Y, X_output, Y_output, 
dummy_list, scoring, best_score, best_param, SMOTE, imbalance_ratio):

    scoring_method_dict = {"accuracy": accuracy_score, "precision": precision_score, "recall": recall_score, "f1": f1_score}
    scoring_method_dict_proba = {"roc_auc": roc_auc_score, "average_precision": average_precision_score}

    try:
        X_columns = list(x_test.columns)
    except:
        pass
        
    #For the convenience the versus report should be separated:
    if DoVersus is False:
        if num == 0:
            summary = open("Summary.txt", "w")
        else:
            summary = open("Summary.txt", "a")
    else:
        if num == 0:
            summary = open("Summary_Versus.txt", "w")
        else:
            summary = open("Summary_Versus.txt", "a")

    summary.write("#############################################################################\n")
    summary.write(str(functional_group))
    summary.write("\n")
    #Because some models like GaussianNB don't have best params:
    if best_score is not None and best_param is not None:
        summary.write(f"The scoring method for determining the best parameters was {scoring} scoring\n")
        summary.write("Grid best score: ")
        summary.write(str(best_score))
        summary.write("\n")
        summary.write("Grid best parameter: ")
        summary.write(str(best_param))
        summary.write("\n")
        
    ########################
    Y_list = list(Y[functional_group])
    ########################
    
    # Now it's time for train/test scores:
    grid_svc_fit = grid_svc.fit(x_train, y_train)

    #################################################################################
    y_predict = grid_svc_fit.predict(x_train)
    summary.write("\n")
    summary.write("The scores for train set:\n")
    summary.write(str(confusion_matrix(y_train, y_predict)))
    scores_dict = {}
    for scoring_method_name in scoring_method_dict:
        scoring_method = scoring_method_dict[scoring_method_name]
        the_score = scoring_method(y_train, y_predict)
        scores_dict[scoring_method_name] = the_score
    try:

        # The probability for the "1" value is in the first column.
        # For the ROC AUC metric, it's necessary to predict the probability scores for each class
        # rather than the actual classification labels.
        # The ROC curve is created by varying the classification threshold, and the AUC score is
        # calculated by measuring the area under the ROC curve.
        # Consequently, a higher AUC score indicates that the model is better at distinguishing
        # between positive and negative samples.

        y_predict_proba = grid_svc_fit.predict_proba(x_train)[:, 1]
        
        
        for scoring_method_name in scoring_method_dict_proba:
            scoring_method = scoring_method_dict_proba[scoring_method_name]
            the_score = scoring_method(y_train, y_predict_proba)
            scores_dict[scoring_method_name] = the_score
    except:
        for scoring_method_name in scoring_method_dict_proba:
            the_score = "No probability available"
            scores_dict[scoring_method_name] = the_score
        
    for score_name in scores_dict:
        summary.write("\n")
        summary.write(f"{score_name} score is: {scores_dict[score_name]}")
    summary.write("\n\n")
    #################################################################################
        
        
    # The test score:
    #################################################################################
    y_predict = grid_svc_fit.predict(x_test)
    summary.write("The scores for test set:\n")
    summary.write(str(confusion_matrix(y_test, y_predict)))
    scores_dict = {}
    for scoring_method_name in scoring_method_dict:
        scoring_method = scoring_method_dict[scoring_method_name]
        the_score = scoring_method(y_test, y_predict)
        scores_dict[scoring_method_name] = the_score
        
    try:
        y_predict_proba = grid_svc_fit.predict_proba(x_test)[:, 1]
        for scoring_method_name in scoring_method_dict_proba:
            scoring_method = scoring_method_dict_proba[scoring_method_name]
            the_score = scoring_method(y_test, y_predict_proba)
            scores_dict[scoring_method_name] = the_score
    except:
        for scoring_method_name in scoring_method_dict_proba:
            the_score = "No probability available"
            scores_dict[scoring_method_name] = the_score
        
    for score_name in scores_dict:
        summary.write("\n")
        summary.write(f"{score_name} score is: {scores_dict[score_name]}")
    summary.write("\n\n")
    #################################################################################

    try:
        #Feature importance if available:
        grid_svc_fit = grid_svc.fit(x_train, y_train)
        feature_importance_list = list(grid_svc_fit.feature_importances_)

        #Getting the columns:
        x_columns = x_train.columns
        importance_dict = {"X_columns": [], "Importance": []}
        index = -1
        for importance_value in feature_importance_list:
            index += 1
            importance_dict["X_columns"].append(x_columns[index])
            importance_dict["Importance"].append(importance_value)

        importance_df = pd.DataFrame(importance_dict)
        importance_df.sort_values(by="Importance", inplace=True, ascending=False)

        #now let's get the first 6
        first_6_X = list(importance_df["X_columns"].iloc[0:6])
        first_6_importance = list(importance_df["Importance"].iloc[0:6])
        summary.write("\nThe first ten important features: ")
        summary.write("[")
        for thing in first_6_X:
            summary.write(str(thing))
            summary.write(", ")
        summary.write("]")
        summary.write("\n")

        summary.write("The importance value: ")
        summary.write("[")
        for thing in first_6_importance:
            summary.write(str(thing))
            summary.write(", ")
        summary.write("]")
        summary.write("\n")
    except:
        pass
    
    # The Dummies:
    #################################################################################
    summary.write("The dummies scores on test set:\n")
    strategy_list = ["most_frequent", "stratified", "uniform"]
    index = -1
    for dummy in dummy_list:
        index += 1
        dummy_strategy = strategy_list[index]
        y_predict = dummy.predict(x_test)
        
        summary.write(f"The {dummy_strategy} dummy:\n")
        summary.write(str(confusion_matrix(y_test, y_predict)))
        scores_dict = {}
        for scoring_method_name in scoring_method_dict:
            scoring_method = scoring_method_dict[scoring_method_name]
            the_score = scoring_method(y_test, y_predict)
            scores_dict[scoring_method_name] = the_score
            
        try:
            y_predict_proba = grid_svc_fit.predict_proba(x_test)[:, 0]
            
            for scoring_method_name in scoring_method_dict_proba:
                scoring_method = scoring_method_dict[scoring_method_name]
                the_score = scoring_method(y_test, y_predict_proba)
                scores_dict[scoring_method_name] = the_score
        except:
            for scoring_method_name in scoring_method_dict_proba:
                the_score = "No probability available"
                scores_dict[scoring_method_name] = the_score
            
        for score_name in scores_dict:
            summary.write("\n")
            summary.write(f"{score_name} score is: {scores_dict[score_name]}")
        summary.write("\n")
    #################################################################################
    
    summary.write("\n")
    summary.write("The Cross validation:\n")
    #Cross validation:
    #################################################################################
    #cloning first is very important, don't forget
    clone(grid_svc)
    for scoring_method_name in scoring_method_dict:
        
        summary.write(f"{scoring_method_name}:\n")
        
        if SMOTE is False:
            strat_k_fold = StratifiedKFold(n_splits=3, shuffle=True)
            cross_val = cross_val_score(grid_svc, X_output, Y_output[functional_group].values.ravel(), cv=strat_k_fold, scoring=scoring_method_name)
            
            summary.write(f"{[num for num in cross_val]}")
            summary.write("\n")
            
        else:

            # For SMOTE, only the metric values for the test set are provided.
            # As K-fold learning processes the data while disregarding past data, we will not use it in this case.
            # First, let's retrain the model on the SMOTE'd train data.

            x_train = np.array(x_train)
            y_train = np.array(y_train)
            y_test = np.array(y_test)
            grid_svc_fit = grid_svc.fit(x_train, y_train)
            
            #Now the single prediction on the imbalanced data
            y_predict = grid_svc_fit.predict(np.array(x_test))

            #Calculating the score
            scoring_method = scoring_method_dict[scoring_method_name]
            num = scoring_method(y_test, y_predict)

            summary.write(f"{[num]}")
            summary.write("\n")

            # Checking if SMOTE works:
            y_train = list(y_train)
            y_test = list(y_test)
            y_train_new = [num for num in y_train if num == 1]
            y_test_new = [num for num in y_test if num == 1]
            print((len(y_train) - len(y_train_new), len(y_train_new)))
            print((len(y_test) - len(y_test_new), len(y_test_new)))

    try:
        for scoring_method_name in scoring_method_dict_proba:
            summary.write(f"{scoring_method_name}:\n")
            
            if SMOTE is False:
                strat_k_fold = StratifiedKFold(n_splits=3, shuffle=True)
                cross_val = cross_val_score(grid_svc, X_output, Y_output[functional_group].values.ravel(), cv=strat_k_fold, scoring=scoring_method_name)
                                            
                summary.write(f"{[num for num in cross_val]}")
                summary.write("\n")
                
            else:

                # For SMOTE, only the metric values for the test set are provided.
                # As K-fold learning processes the data while disregarding past data, we will not use it in this case.
                # First, let's retrain the model on the SMOTE'd train data.

                x_train = np.array(x_train)
                y_train = np.array(y_train)
                y_test = np.array(y_test)
                grid_svc_fit = grid_svc.fit(x_train, y_train)

                # Now the single prediction on the imbalanced data
                y_predict = grid_svc_fit.predict(np.array(x_test))

                # Calculating the score
                scoring_method = scoring_method_dict[scoring_method_name]
                num = scoring_method(y_test, y_predict)

                summary.write(f"{[num]}")
                summary.write("\n")

               
                
    except:
        summary.write("No probability available\n")
    
    #let's skip the dummies for now. It's clear they're much worse than the model, sooo...        

    summary.write("\n\n")
    
    # Let's inlcude more easily the numer of one and zeros and the sum here:
    summary.write(f"The number of ones: {Y_list.count(1)}")
    summary.write("\n")
    summary.write(f"The number of zeros: {Y_list.count(0)}")
    summary.write("\n")
    summary.write(f"Total number of molecules: {Y_list.count(1) + Y_list.count(0)}")
    summary.write("\n")
    summary.write(f"The imbalance ratio: {imbalance_ratio}")
    summary.write("\n")
    
    summary.close()
        
 
