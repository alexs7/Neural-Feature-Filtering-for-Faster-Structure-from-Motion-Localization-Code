# Added 18/08/2022
names_dict = {
    "C-R" : "Classifier w/ top 10% matches", #0
    "C-All-R" : "Classifier using all matches",  #1
    "C&R-I-R" : "Classifier and Regressor w/ image score", #2
    "C&R-S-R" : "Classifier and Regressor w/ score per session", #3
    "C&R-V-R" : "Classifier and Regressor w/ visibility score",  #4
    "R-I-R" : "Regressor w/ score per image",   #5
    "R-S-R" : "Regressor w/ score per session",  #6
    "R-V-R" : "Regressor w/ visibility score",  #7
    "CM-I-R" : "Combined w/ score per image",  #8
    "CM-S-R" : "Combined w/ score per session",  #9
    "CM-V-R" : "Combined w/ visibility score",  #10
    "C&R-I-R*" : "Class. and Regr. w/ score per image, dist. RANSAC",  #11
    "C&R-S-R*" : "Class. and Regr. w/ score per session, R*",  #12
    "C&R-V-R*" : "Class. and Regr. w/ visibility score, dist. RANSAC",  #13
    "R-I-P" : "Regressor w/ score per image, PROSAC",  #14
    "R-S-P" : "Regressor w/ score per session, PROSAC",  #15
    "R-V-P" :  "Regressor w/ visibility score, PROSAC",  #16
    "CM-I-P" : "Combined w/ score per image, PROSAC",  #17
    "CM-S-P" : "Combined w/ score per session, PROSAC",  #18
    "CM-V-P" : "Combined w/ visibility score, PROSAC",  #19
    "Random" : "Random feature case",  #20
    "Baseline" : "Baseline using all features",  #21
    "RF-PM": "Random Forest Predicting Matchability" #22
}