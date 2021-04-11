# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 11:26:55 2020

@author: Lucas
"""


import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os

def sexint(series):
    
    list_sex = np.zeros_like(series)
    list_sex[np.where(np.array(series) =="male")] = 1
    list_sex[np.where(np.array(series) =="female")] = 0
    
    return list_sex

### transform series to  continue variable
def Embarkedint(series):
    
    list_embarked = np.zeros_like(series)
    list_embarked[np.where(np.array(series) =="Q")] = 0
    list_embarked[np.where(np.array(series) =="C")] = 1
    list_embarked[np.where(np.array(series) =="S")] = 2
    
    
    
    return list_embarked



### make label function
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        axes.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom')



def graph_age_fare(data):
    fig, axes = plt.subplots()

    axes.set_ylabel('Fare')
    axes.set_title('Age x Fare')
    axes.set_xlabel("Age")
    plot_tt = axes.scatter(data.iloc[:,5], data.iloc[:,9], c=data["Survived"])
    a,labels= plot_tt.legend_elements()
    axes.legend(a, ["Dead","Alive"])
    fig.tight_layout()
    fig.savefig(os.path.join("graph_desc","graph_age_fare.png"))

def graph_sex_surv(data_descrip_survived):
    CT_survived_class = pd.crosstab(data["Pclass"], data["Survived"])
    range_ = np.arange(len(Counter(data["Pclass"])))
    width = 0.35
    fig, axes = plt.subplots()
    rects1 = axes.bar(range_ - width/2, CT_survived_class[0], width, label='dead')
    rects2 = axes.bar(range_ + width/2, CT_survived_class[1], width, label='a live')

    axes.set_ylabel('Scores')
    axes.set_title('Scores by Pclass and survived')
    axes.set_xticks(range_)

    axes.set_autoscaley_on(True)
    axes.set_xticklabels(np.sort(list(Counter(data["Pclass"]).keys())))
    axes.legend()
    axes.set_ylim((0,600))

    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    fig.savefig(os.path.join("graph_desc","bar_sex_surv.png"))

def graph_pie_Pclass (data):
    fig, axes = plt.subplots()
    freq_list = pd.Series(list(Counter(data["Pclass"]).values()))/sum(Counter(data["Pclass"]).values())
    plt.pie(freq_list, labels = Counter(data["Pclass"]).keys(),autopct = '%1.1f%%', explode = (0,0.1,0), colors = ['#1f77b4', '#e377c2', '#2ca02c'])
    plt.title("Distribution of Pclass")
    fig.savefig(os.path.join("graph_desc","pie_Pclass.png"))

def graph_distrib_Fare(data):
    fig, axes = plt.subplots()
    plt.hist(data["Fare"], density = False, cumulative = False, align = "mid")
    plt.xlabel("Fare")
    plt.title("Distribution of Fare")
    fig.savefig(os.path.join("graph_desc","distrib_Fare.png"))


def graph_distrib_sex(data):
    fig, axes = plt.subplots()
    freq_sex = pd.Series(list(Counter(data["Sex"]).values()))/sum(Counter(data["Sex"]).values())
    plt.pie(freq_sex, labels = ["male", "female"],autopct = '%1.1f%%', explode = (0,0.1), colors = ['#1f77b4', '#e377c2', '#2ca02c'])
    plt.title("Distribution of Sex")
    fig.savefig(os.path.join("graph_desc","distrib_sex.png"))

def graph_distrib_Sibsp(data):
    fig, axes = plt.subplots()
    plt.bar(Counter(data["SibSp"]).keys(),Counter(data["SibSp"]).values(), orientation = "vertical", color = '#e377c2', edgecolor = '#2ca02c')
    plt.title("Distribution of SibSp")
    plt.xlabel("SibSp")
    fig.savefig(os.path.join("graph_desc","distrib_Sibsp.png"))

def graph_score_Pclass_sex(data):
    CT_sex_class = pd.crosstab(data["Pclass"], data["Sex"])
    range_ = np.arange(len(Counter(data["Pclass"])))
    width = 0.35
    fig, axes = plt.subplots()

    axes.set_ylabel('Scores')
    axes.set_title('Scores by Pclass and gender')
    axes.set_xticks(range_)
    axes.set_xticklabels(Counter(data["Pclass"]).keys())

    axes.set_ylim((0,550))

    rects1 = axes.bar(range_ - width/2, CT_sex_class["male"], width, label='Men')
    rects2 = axes.bar(range_ + width/2, CT_sex_class["female"], width, label='Women')

    autolabel(rects1)
    autolabel(rects2)
    axes.legend([rects1,rects2], ["male","female"])
    fig.savefig(os.path.join("graph_desc","bar_sex_class.png"))


def graph_sex_class(data):
    ## pie cross 2 var (problems legend)
    fig, ax = plt.subplots()
    size = 0.3
    vals = np.array([CT_sex_class.iloc[0],CT_sex_class.iloc[1] , CT_sex_class.iloc[2]])
    cmap = plt.get_cmap("tab20c")
    outer_colors = cmap(np.arange(3)*4)
    inner_colors = cmap(np.array([1,2,5,6,8,9]))

    ax.pie(vals.sum(axis=1), radius=1, colors=outer_colors,
            wedgeprops=dict(width=size, edgecolor='w'))
    ax.pie(vals.flatten(), radius=1-size, colors=inner_colors,
            wedgeprops=dict(width=size, edgecolor='w'))
    ax.set(aspect="equal", title='Pie plot with `ax.pie`')
    ax.legend([1,2,3])
    fig.savefig(os.path.join("graph_desc","graph_sex_class.png"))

