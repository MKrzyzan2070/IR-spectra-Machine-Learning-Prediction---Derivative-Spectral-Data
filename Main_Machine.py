import Mining.Making_the_json as Make_json
import Mining.Finishing_touch as finish

######################################################################
######################################################################

# COMMENT THIS OUT IF YOU DO NOT WANT TO USE IT!
Make_json.making_json("gas", 800, 3600, 10)
finish.lets_put_a_smile_on_your_face()

######################################################################
######################################################################

import The_machine_proper.The_Project_Machine as MACHINE

MACHINE.the_machine(Create_CSV=True, Only_Create_CSV=False,
                    OneVsAll=True,
                    DoPlain=True, DoDerivative=True,
                    Categorical=False, ThreshThresh=None, DerivativeThresh=None, NumericalCorrection=False,
                    Snipping=False, Class_Weight=False, SMOTE=False,
                    Scoring="f1",
                    OneD_Mixing=False, OneD_Mixing_No_Other_Group=False, MixOnly=False, DoVersus=False,
                    SVM=False, Logistic=False, KNNSingle=True, GaussianNB=False, ColdForestSingle=False, ColdMLP=False, GradientBoost=False,
                    WarmForestSingle=False, Perceptron=False, WarmMLP=False,
                    KNNMulti=False)


