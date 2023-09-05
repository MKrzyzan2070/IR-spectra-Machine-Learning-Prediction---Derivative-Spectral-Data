import pandas as pd
import json
pd.options.mode.chained_assignment = None

def the_machine(Create_CSV, Only_Create_CSV,
                OneVsAll,
                DoPlain, DoDerivative,
                Categorical, ThreshThresh, DerivativeThresh, NumericalCorrection,
                Snipping, Class_Weight, SMOTE, 
                Scoring,
                OneD_Mixing, OneD_Mixing_No_Other_Group, MixOnly, DoVersus,
                SVM, Logistic, KNNSingle, GaussianNB, ColdForestSingle, ColdMLP, GradientBoost,
                WarmForestSingle, Perceptron, WarmMLP,
                KNNMulti):

    import The_machine_proper.Making_the_final_csv_file_that_would_go_to_machine as make_csv
    from The_machine_proper import Machine_Engine

    file = "stop_warning"
    if Create_CSV is True:
        # Ok so now the uniform and normalized csv needs to be created:
        # And yeah this is done regardless of the Truth/False value of the DoPlain
        
        print("Making the basic CSV file")
        plain_csv_signal_column_list = make_csv.Creating_Uniform_Plain_CSV()

        #if DoDerivative is present, then we need to create the derivative csv:
        if DoDerivative is True:
            print("Making the derivative CSV file")
            derivative_csv_column_list = make_csv.Creating_Uniform_Derivative_CSV(plain_csv_signal_column_list)
           
            
        # Now for the numerical showdown. This has to happened regardless of Categorical value
        if DoPlain is True and DoDerivative is False:
            file = "Uniform_csv_plain"
            
        elif DoPlain is False and DoDerivative is True:
            file = "Uniform_csv_derivative"
            
        elif DoPlain is True and DoDerivative is True:
            make_csv.Make_IR_Numerical_Both()
            file = "Uniform_csv_both"
            
        else:
            print("DoPlain and DoDerivative can't be both False")
            exit()
            
        #Now that the numerical base of the csv files is complete it's time for the categorical data:
        if Categorical is True:
            #All the files paths and whatnot are within the reams of make_csv.Make_IR_Categorical
            make_csv.Make_IR_Categorical(df_file=file, ThreshThresh=ThreshThresh, DerivativeThresh=DerivativeThresh, NumericalCorrection=NumericalCorrection)
            
            #It's going to be the Final_csv already !
            file = "Final_csv"


        ########### @@@
        # The file name needs to be stored
        json_object = json.dumps(file)
        with open("The_machine_proper/CSV_file_name.json", "w") as doc:
            doc.write(json_object)
        ########### @@@
        
        
    # reading the name of the file:
    with open("The_machine_proper/CSV_file_name.json", 'r') as openfile:
        file = json.load(openfile)
        
    ##################################
    print("Assigning the functional groups from the SMILE string...")
    output = make_csv.make_bonds_from_smile(file)
    print("Assigning complete")
    ###################################

    X_columns = output[0]
    Y_columns = output[1]
    
    if Only_Create_CSV is True:
        pass
        
    else:

        ##################################
        print("The Machine begins...")
        # Here we have the Final csv file which was created during the make_bonds_from_smile step
        df = pd.read_csv("The_machine_proper/Final_csv.csv")
        ##################################

        if OneVsAll is True:
            #The Y will go one at a time

            if OneD_Mixing is False:
                ############ The engine for 1D array, going one by one
                Summary_before = False
                Machine_Engine.oneD_array_engine(df, Y_columns, X_columns, Snipping, Scoring,
                              SVM, Logistic, ColdForestSingle, ColdMLP, KNNSingle, GaussianNB, GradientBoost, WarmForestSingle, Perceptron, WarmMLP, Class_Weight, SMOTE)

            elif OneD_Mixing is True:
                Y_Mixlist = [["Ketone1", "Ketone2", "Ketone3"]]

                Machine_Engine.oneD_array_engine_Mix(MixOnly, OneD_Mixing_No_Other_Group, df, Y_columns, Y_Mixlist, X_columns, Snipping, Scoring,
                                      SVM, Logistic, ColdForestSingle, ColdMLP, KNNSingle, GaussianNB, GradientBoost, WarmForestSingle, Perceptron, WarmMLP, Class_Weight, SMOTE)
            if DoVersus is True:
                Versus1 = [["Ketone1"]]
                Versus2 = [["Ketone2"]]


                Machine_Engine.Do_The_Versus(df, Versus1, Versus2, X_columns, Snipping, Scoring,
                          SVM, Logistic, ColdForestSingle, ColdMLP, KNNSingle, GaussianNB, GradientBoost, WarmForestSingle, Perceptron, WarmMLP, Class_Weight, SMOTE)

        ############################################################################################################################
        ############################################################################################################################







        ################################################################################
        ################################################################################
        ################## Yet to be built:
        else:
            # Here the total multi classification will be created. But how
            # For now let's opt for the KNN. hopefully it will work
            Machine_Engine.MultiLabel(df, Y_columns, X_columns, Scoring, KNNMulti)

        ################################################################################
        ################################################################################