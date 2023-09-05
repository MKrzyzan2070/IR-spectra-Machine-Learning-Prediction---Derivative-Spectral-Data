import Mining.Making_the_json as Make_json
import Mining.Finishing_touch as finish
import The_machine_proper.The_Project_Machine as MACHINE
import os
import numpy as np
import time

# The range:
range_dict = {"from": list(np.linspace(500, 1600, num=12)),
              "to": list(np.linspace(2200, 3800, num=17))}

# The state:
state_list = ["gas", "liquid", "solid", "solid & liquid", "liquid & solid", "solid & gas",
              "gas & solid", "gas & liquid", "liquid & gas", "all"]


####################################################

# NA TEN MOMENT:
################## @@@
# The state:
state_list = ["gas", "liquid", "solid"]

#Na ten moment:
range_dict = {"from": [600],
              "to": [3600]}
################## @@@

# The resolution:
resolution_list = list(range(20, 0, -2))

# The plain:
DoPlain_list = [False, True]

# The derivative:
DoDerivative_list = [False, True]

# The noise correction:
Noise_dict = {"Categorical": [False],
              "ThreshThresh": [None],
              "DerivativeThresh": [None],
              "NumericalCorrection": [False]}

# The class weight:
class_weight_list = [False]

#SMOTE:
SMOTE_list = [False]

# The model:
model_list = ["Logistic", "KNNSingle"]

for range_from in range_dict["from"]:
    for range_to in range_dict["to"]:
        for resolution in resolution_list:
            for state in state_list:
                
                prep_time_1_start = time.time()
                ################################################
                Make_json.making_json(state, range_from, range_to, resolution)
                finish.lets_put_a_smile_on_your_face()
                ################################################
                prep_time_1_stop = time.time()
                
                prep_time_1 = prep_time_1_stop - prep_time_1_start
                
                for derivative in DoDerivative_list:
                    for plain in DoPlain_list:
                        for index in range(len(Noise_dict["Categorical"])):
               
                            #######################################
                            Categorical = Noise_dict["Categorical"][index]
                            
                            if Categorical is True:
                                ThreshThresh = Noise_dict["ThreshThresh"][index]
                                DerivativeThresh = Noise_dict["DerivativeThresh"][index]
                                NumericalCorrection = Noise_dict["NumericalCorrection"][index]
                            else:
                                ThreshThresh = None
                                DerivativeThresh = None
                                NumericalCorrection = False
                            #######################################
               
                                
                            if plain is True or derivative is True:
                            
                                prep_time_2_start = time.time()
                                ##########################################################################################
                                MACHINE.the_machine(Create_CSV=True, Only_Create_CSV=True,
                                                    OneVsAll=True,
                                                    DoPlain=plain, DoDerivative=derivative,
                                                    Categorical=Categorical, ThreshThresh=ThreshThresh,
                                                    DerivativeThresh=DerivativeThresh,
                                                    NumericalCorrection=NumericalCorrection,
                                                    Snipping=False, Class_Weight=False, SMOTE=False,
                                                    Scoring="f1",
                                                    OneD_Mixing=False, OneD_Mixing_No_Other_Group=False, MixOnly=False,
                                                    DoVersus=False,
                                                    SVM=False, Logistic=False, KNNSingle=False, GaussianNB=True,
                                                    ColdForestSingle=False, ColdMLP=False, GradientBoost=False,
                                                    WarmForestSingle=False, Perceptron=False, WarmMLP=False,
                                                    KNNMulti=False)
                                                
                                                
                                ###############################################################################################
                                prep_time_2_stop = time.time()
                                prep_time_2 = prep_time_2_stop - prep_time_2_start

                                for Class_weight in class_weight_list:

                                    ######################################
                                    for model in model_list:
                                        # @@@@@@@@@@@@@@@@@@@@
                                        SVM = False
                                        Logistic = False
                                        ColdForestSingle = False
                                        KNNSingle = False
                                        ColdMLP = False
                                        # @@@@@@@@@@@@@@@@@@@@
                                        
                                        if model == "SVM":
                                            SVM = True
                                        elif model == "Logistic":
                                            Logistic = True
                                        elif model == "ColdForestSingle":
                                            ColdForestSingle = True
                                        elif model == "KNNSingle":
                                            KNNSingle = True
                                        elif model == "ColdMLP":
                                            ColdMLP = True
                                    ######################################
                                        
                                        
                                        ### @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                                        # This enetiery custom honestly:
                                        if plain is True:
                                            plain_deriv = "PlainOnly"
                                        if derivative is True:
                                            plain_deriv = "DerivativeOnly"
                                        if plain is True and derivative is True:
                                            plain_deriv = "Both"
                                        
                                        new_directory = f"{model}_{plain_deriv}"
                                        
                                        if os.path.exists(f"Summaries/{new_directory}") is False:
                                            os.mkdir(f"Summaries/{new_directory}")
                                        ### @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                                        
                                        for SMOTE in SMOTE_list:
                                        
                                            if not (SMOTE is True and Categorical is True):

                                                for num in range(5):
                                                    work_time_start = time.time()

                                                    if plain is True or derivative is True:
                                                        MACHINE.the_machine(Create_CSV=False, Only_Create_CSV=False, 
                                                                            OneVsAll=True,
                                                                            DoPlain=plain, DoDerivative=derivative,
                                                                            Categorical=Categorical, ThreshThresh=ThreshThresh, DerivativeThresh=DerivativeThresh,
                                                                            NumericalCorrection=NumericalCorrection,
                                                                            Snipping=False, Class_Weight=Class_weight, SMOTE=SMOTE,
                                                                            Scoring="f1",
                                                                            OneD_Mixing=False, OneD_Mixing_No_Other_Group=False, MixOnly=False,
                                                                            DoVersus=False,
                                                                            SVM=SVM, Logistic=Logistic, KNNSingle=KNNSingle, GaussianNB=False,
                                                                            ColdForestSingle=ColdForestSingle, ColdMLP=ColdMLP, GradientBoost=False,
                                                                            WarmForestSingle=False, Perceptron=False, WarmMLP=False,
                                                                            KNNMulti=False)

                                                        summary_path_name = f"Summaries/{new_directory}/Summary" +f"{range_from}_{range_to}_" \
                                                                                                 f"{resolution}_{state}_{plain}_{derivative}_{Categorical}_"\
                                                                                                 f"{ThreshThresh}_{DerivativeThresh}" \
                                                                                                 f"_{Class_weight}_{model}" + str(num+1) + ".txt"
                                                                                                 
                                                                                                 
                                                        os.rename("Summary.txt", summary_path_name)

                                                        work_time_stop = time.time()

                                                        work_time = work_time_stop - work_time_start

                                                        with open(summary_path_name, "a") as summary:
                                                            summary.write("The time values are the following:\n")
                                                            summary.write(f"Json preparation time: {prep_time_1}\n")
                                                            summary.write(f"CSV preparation time: {prep_time_2}\n")
                                                            summary.write(f"Working time: {work_time}\n")
                                                            summary.write("\n\n")
                                                            summary.write("THE VARIABLES:\n")
                                                            summary.write(f"The range is from: {range_from} to: {range_to}\n")
                                                            summary.write(f"Resolution: {resolution}\n")
                                                            summary.write(f"The state is: {state}\n")
                                                            summary.write(f"Plain: {plain}\n")
                                                            summary.write(f"Derivative: {derivative}\n")
                                                            summary.write(f"Anti noise applied to derivative: {DerivativeThresh}\n")
                                                            summary.write(f"Class weight: {Class_weight}\n")
                                                            summary.write(f"The model: {model}\n")




