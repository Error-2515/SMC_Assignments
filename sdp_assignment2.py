# -*- coding: utf-8 -*-
"""SDP_Assignment2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1R1ZnFjS0UwaQzNsVoUarFjolWzT2KBSc
"""

import pandas as pd
data=pd.read_csv("https://raw.githubusercontent.com/Insiya24/datasets/main/WA_Fn-UseC_-HR-Employee-Attrition.csv")

data

data.head()

data.dtypes

datagrp=data.groupby(["DistanceFromHome"])[["MonthlyIncome"]]

datagrp.mean()

data.dtypes

data["Department"].mode()

data["EducationField"].mode()

data["EmployeeCount"].mean()

data["EmployeeNumber"].mean()

data["EnvironmentSatisfaction"].mean()

data["Gender"].mode()

data["HourlyRate"].mean()

data["JobInvolvement"].mean()

data["JobLevel"].mean()

data["JobSatisfaction"].mean()

data["PercentSalaryHike"].mean()

data["Department"].fillna("Research & Development",inplace=True)
data["EducationField"].fillna("Life Sciences",inplace=True)
data["EmployeeCount"].fillna(1.0,inplace=True)
data["EmployeeNumber"].fillna(1026.051,inplace=True)
data["EnvironmentSatisfaction"].fillna(2.722,inplace=True)
data["Gender"].fillna("Male",inplace=True)
data["HourlyRate"].fillna(65.88,inplace=True)
data["JobInvolvement"].fillna(2.73,inplace=True)
data["JobLevel"].fillna(2.06,inplace=True)
data["JobSatisfaction"].fillna(2.72,inplace=True)
data["PercentSalaryHike"].fillna(15.22,inplace=True)

data.isna().sum()

data.dtypes

from sklearn import preprocessing
le= preprocessing.LabelEncoder()

data["Attrition"]=le.fit_transform(data["Attrition"])
data["Department"]=le.fit_transform(data["Department"])
data["EducationField"]=le.fit_transform(data["EducationField"])
data["Gender"]=le.fit_transform(data["Gender"])
data["MaritalStatus"]=le.fit_transform(data["MaritalStatus"])
data["OverTime"]=le.fit_transform(data["OverTime"])

data["OverTime"].value_counts()

data=data.drop(["StandardHours"],axis=1)
# data.drop(["EmployeeCount"],axis=1)

import seaborn
import matplotlib.pyplot as pt
pt.figure(figsize=(13,13))

correlation=data.corr()
heatmap=seaborn.heatmap(correlation,annot=True)


pt.show()

"""Question 2"""





