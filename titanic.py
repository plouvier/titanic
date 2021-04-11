# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 11:26:55 2020

@author: Lucas
"""

########################## import ####################
######################################################


import csv
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix
from module_function import sexint, Embarkedint, autolabel
from module_function import graph_age_fare , graph_sex_surv, graph_pie_Pclass, graph_distrib_Fare, graph_distrib_sex, graph_score_Pclass_sex, graph_distrib_Sibsp , graph_sex_class



######################### main #######################
######################################################

##### load data set

if __name__ == '__main__':                
    os.getcwd()
    os.chdir(r"D:\Kaggle\titanic")

    ### load data

    train_data = pd.read_csv(os.path.join("data","train.csv"), sep = ",", header = 0 )
    test_data = pd.read_csv(os.path.join("data","test.csv"), sep = ",", header = 0 )
    survived_test = pd.read_csv(os.path.join("data","gender_submission.csv"), sep = ",", header = 0 )
    y_train = train_data["Survived"]
    y_test = survived_test["Survived"]

    test_data2 = pd.concat((test_data, survived_test["Survived"]),axis = 1)
    data_descrip = pd.concat((train_data.drop(columns = ["Survived"]),test_data), ignore_index = True)
    data_descrip_survived = pd.concat((train_data,test_data2), ignore_index = True)


    ###  replace na to 0 (no need more complexe management of na values) 

    train_data.fillna(value = 0, inplace = True)
    test_data.fillna(value = 0, inplace = True)

    ## change some variable

    tempo_sex = sexint(train_data["Sex"])
    train_data["Sex"] = tempo_sex
    tempo_embarked = Embarkedint(train_data["Embarked"])
    train_data["Embarked"] = tempo_embarked

    tempo_sex = sexint(test_data["Sex"])
    test_data["Sex"] = tempo_sex
    tempo_embarked = Embarkedint(test_data["Embarked"])
    test_data["Embarked"] = tempo_embarked


    ######################## description of variables
    ######################## description of variables crossed

    graph_age_fare(data_descrip_survived)
    """
    graph_age_fare(data_descrip_survived)
    graph_sex_surv(data_descrip_survived)
    graph_pie_Pclass (data_descrip)
    graph_distrib_Fare(data_descrip)
    graph_distrib_sex(data_descrip)
    graph_distrib_Sibsp(data_descrip)
    graph_score_Pclass_sex(data_descrip)
    graph_sex_class(data_descrip)

    ### extreme value Fare

    extr_fare = list(np.where(np.array(data_descrip["Fare"])>400))
    extr_fare_desc = []
    for i in extr_fare:
        extr_fare_desc.append((data_descrip["Age"].loc[i], data_descrip["Pclass"].loc[i], data_descrip["Sex"].loc[i], data_descrip["SibSp"].loc[i], data_descrip["Parch"].loc[i]))

    """

    ###################  MODELE ##########################################

    x_train_keep = train_data.drop(columns =["PassengerId", "Survived", "Name", "Ticket", "Cabin"])
    x_test_keep = test_data.drop(columns =["PassengerId", "Name", "Ticket", "Cabin"])

    ########### Random Forest

    forest = RandomForestClassifier(n_estimators = 70, random_state = 2, max_depth=8,max_features = 6)
    forest.fit(x_train_keep,y_train)
    res_train = forest.score(x_train_keep,y_train)
    res_test = forest.score(x_test_keep,y_test)

    forest_predict = forest.predict(x_test_keep)
    cm_forest = confusion_matrix(y_test, forest_predict)

    sub_forest = pd.DataFrame(np.transpose((test_data["PassengerId"],forest_predict)))
    sub_forest.columns = ["PassengerId","Survived"]
    sub_forest.to_csv(os.path.join("titanic_result","submission_titanic_forest.csv"),columns=sub_forest.columns, index = False, sep = ",")

    ################# standardisation des donées

    data_train_scaled = preprocessing.scale(x_train_keep)
    data_test_scaled = preprocessing.scale(x_test_keep)

    ########### Neuronal Network
    ### avec ce modèle les différents paramètres testés sont le nombre d'hidden layer, la fonction d'activation
    ### le nombre d'iteration max pour la retropropagation (convergence ou non), le solver, 


    ## sans standardisation
    mlp = MLPClassifier(random_state = 3)
    mlp.fit(x_train_keep, y_train)
    mlp.score(x_train_keep, y_train)
    mlp.score(x_test_keep,y_test)


    ## avec standardisation

    mlp = MLPClassifier(random_state = 3, activation = "tanh", hidden_layer_sizes=(100,), )
    mlp.fit(data_train_scaled, y_train)
    res_mlp_train = mlp.score(data_train_scaled, y_train)
    res_mlp_test = mlp.score(data_test_scaled,y_test)

    mlp_prediction = mlp.predict(data_test_scaled)
    cm = confusion_matrix(y_test, mlp_prediction)

    survived_label = ["Dead", "Alive"]

    fig, ax = plt.subplots()
    im = ax.imshow(cm)


    sub = pd.DataFrame(np.transpose((test_data["PassengerId"],mlp_prediction)))
    sub.columns = ["PassengerId","Survived"]
    ####################### Sortie  ############################

    sub.to_csv(os.path.join("titanic_result","submission_titanic.csv"),columns=sub.columns, index = False, sep = ",")












