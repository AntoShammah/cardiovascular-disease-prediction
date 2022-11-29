import numpy as np 
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import messagebox


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


class HeartPrediction:

 def __init__(self,root):
  self.root= root
  root.title("Heart Disease Prediction")
  root.geometry('500x800')
  #Button(root, text='Reset',width=20,bg='brown',fg='white',command=validation).place(x=180,y=430)
  # Centering Root Window on Screen

  w = 800 # width for the Tk root
  h = 500 # height for the Tk root

  # get screen width and height
  ws = root.winfo_screenwidth() # width of the screen
  hs = root.winfo_screenheight() # height of the screen

  # calculate x and y coordinates for the Tk root window
  x = (ws/2) - (w/2)
  y = (hs/2) - (h/2)


  root["bg"] = '#98fb98'
  # set the dimensions of the screen 
  # and where it is placed
  root.geometry('%dx%d+%d+%d' % (w, h, x, y))


  # read data
  self.df = pd.read_csv('heartt.csv')
# data shape
  print(self.df.shape)

# check for dupicates
  duplicate_df = self.df[self.df.duplicated()]
#print(duplicate_df)

# checking for missing values
  self.df.isna().sum()
  null = self.df[self.df.isna().any(axis=1)]
#print(null)

# checking distributions using histograms
##  fig = plt.figure(figsize = (15,20))
##  ax = fig.gca()
##  self.df.hist(ax = ax)


# Dropping all rows with missing data
  self.df = self.df.dropna()
  self.df.isna().sum()
#print(df.columns)

  label_0 = Label(root, text="Heart Disease Prediction",width=30,font=("Courier New", 15, "bold"),bg='#98fb98',fg='red')
  label_0.place(x=180,y=13)

  Button(root, text='Feature importance',width=20,bg='green',fg='white',command=self.feature_imp).place(x=80,y=60)
  Button(root, text='Pipeline Model',width=20,bg='green',fg='white',command=self.Pipemodel).place(x=250,y=60)
  Button(root, text='Algorithm comparision',width=20,bg='green',fg='white',command=self.comparealg).place(x=420,y=60)

  Button(root, text='Modelling',width=20,bg='green',fg='white',command=self.modellingg).place(x=80,y=110)
  Button(root, text='Best Model',width=20,bg='green',fg='white',command=self.bestmodel).place(x=250,y=110)
  Button(root, text='Apply the Model',width=20,bg='green',fg='white',command=self.visi).place(x=420,y=110)


 def visi(self):
  self.age=StringVar()
  self.gender=StringVar()
  self.cigs =StringVar()
  self.sysbp=StringVar()
  self.diabp=StringVar()
  self.totChol=StringVar()
  self.prevalentHyp=StringVar()
  self.diabetes =StringVar()
  self.glucose=StringVar()
  self.BPMeds=StringVar()

  label_1 = Label(root, text="Input Patient Information",width=30,font=("Courier New", 15, "bold"),bg='#98fb98',fg='red')
  label_1.place(x=180,y=160)
  
  c=self.root.register(self.only_numeric_input)
  self.label_1 = Label(self.root, text="Age",width=15,font=("bold", 10),bg='#98fb98', anchor='w')
  self.label_1.place(x=20,y=210)

  self.entry_1 = Entry(self.root,textvar=self.age,validate="key",validatecommand=(c,'%P'))
  self.entry_1.place(x=170,y=210)

  self.label_2 = Label(self.root, text="Gender(Male-1,Female-0)",width=25,font=("bold", 10),bg='#98fb98', anchor='w')
  self.label_2.place(x=320,y=210)

  self.entry_2 = Entry(self.root,textvar=self.gender,validate="key",validatecommand=(c,'%P'))
  self.entry_2.place(x=470,y=210)

  self.label_3 = Label(self.root, text="Cigarettes per Day",width=15,font=("bold", 10),bg='#98fb98', anchor='w')
  self.label_3.place(x=20,y=260)

  self.entry_3 = Entry(self.root,textvar=self.cigs,validate="key",validatecommand=(c,'%P'))
  self.entry_3.place(x=170,y=260)

  self.label_4 = Label(self.root, text="Systolic BP",width=15,font=("bold", 10),bg='#98fb98', anchor='w')
  self.label_4.place(x=320,y=260)

  self.entry_4 = Entry(self.root,textvar=self.sysbp,validate="key",validatecommand=(c,'%P'))
  self.entry_4.place(x=470,y=260)

  self.label_5 = Label(self.root, text="Diastolic BP",width=15,font=("bold", 10),bg='#98fb98', anchor='w')
  self.label_5.place(x=20,y=310)

  self.entry_5 = Entry(self.root,textvar=self.diabp,validate="key",validatecommand=(c,'%P'))
  self.entry_5.place(x=170,y=310)

  self.label_6 = Label(self.root, text="Cholesterin Level",width=15,font=("bold", 10),bg='#98fb98', anchor='w')
  self.label_6.place(x=320,y=310)

  self.entry_6 = Entry(self.root,textvar=self.totChol,validate="key",validatecommand=(c,'%P'))
  self.entry_6.place(x=470,y=310)

  self.label_7 = Label(self.root, text="Hypertensive(Y-1,N-0)",width=20,font=("bold", 10),bg='#98fb98', anchor='w')
  self.label_7.place(x=20,y=360)

  self.entry_7 = Entry(self.root,textvar=self.prevalentHyp,validate="key",validatecommand=(c,'%P'))
  self.entry_7.place(x=170,y=360)

  self.label_8 = Label(self.root, text="Diabetes(Y-1,N-0)",width=15,font=("bold", 10),bg='#98fb98', anchor='w')
  self.label_8.place(x=320,y=360)

  self.entry_8 = Entry(self.root,textvar=self.diabetes,validate="key",validatecommand=(c,'%P'))
  self.entry_8.place(x=470,y=360)

  self.label_9 = Label(self.root, text="Glucose",width=15,font=("bold", 10),bg='#98fb98', anchor='w')
  self.label_9.place(x=20,y=410)

  self.entry_9 = Entry(self.root,textvar=self.glucose,validate="key",validatecommand=(c,'%P'))
  self.entry_9.place(x=170,y=410)

  self.label_10 = Label(self.root, text="BPMedication(Y-1,N-0)",width=20,font=("bold", 10),bg='#98fb98', anchor='w')
  self.label_10.place(x=320,y=410)

  self.entry_10 = Entry(self.root,textvar=self.BPMeds,validate="key",validatecommand=(c,'%P'))
  self.entry_10.place(x=470,y=410)

  Button(self.root, text='Submit',width=15,bg='brown',fg='white',command=self.applymod).place(x=250,y=460)

 def applymod1(self):
  parameters=['sysBP', 'glucose','age','totChol','cigsPerDay','diaBP','prevalentHyp','diabetes','BPMeds','male']
  new_features=self.df[['sysBP', 'glucose','age','totChol','cigsPerDay','diaBP','prevalentHyp','diabetes','BPMeds','male','TenYearCHD']]
  x=new_features.iloc[:,:-1]
  y=new_features.iloc[:,-1]
  x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=5)
  logreg=LogisticRegression()
  logreg.fit(x_train,y_train)
  
  y_pred=logreg.predict(x_test)
  my_df = x.iloc[30:40,:]
  print(new_features.iloc[30:40,:])
  my_y_pred = logreg.predict(my_df)
  print('\n')
  print('Result:')
  print(my_y_pred)

  

 def applymod(self):
  my_predictors = []
  parameters=['sysBP', 'glucose','age','totChol','cigsPerDay','diaBP','prevalentHyp','diabetes','BPMeds','male']
  my_predictors.append(self.sysbp.get())
  my_predictors.append(self.glucose.get())
  my_predictors.append(self.age.get())
  my_predictors.append(self.totChol.get())
  my_predictors.append(self.cigs.get())
  my_predictors.append(self.diabp.get())
  my_predictors.append(self.prevalentHyp.get())
  my_predictors.append(self.diabetes.get())
  my_predictors.append(self.BPMeds.get())
  my_predictors.append(self.gender.get())
  print(my_predictors)
  my_data = dict(zip(parameters, my_predictors))

  my_df = pd.DataFrame(my_data, index=[0])
  print(my_df)
  scaler = MinMaxScaler(feature_range=(0,1)) 
   
  # assign scaler to column:
  #my_df_scaled = pd.DataFrame(scaler.fit_transform(my_df), columns=my_df.columns)
  my_y_pred = self.dtc_up.predict(my_df)
  print('\n')
  print('Result:')
  if my_y_pred == 1:
      print("The patient has Heart Disease.")
  if my_y_pred == 0:
      print("The patient has not Heart Disease.")
    

 def only_numeric_input(self,e):
  #this is allowing all numeric input
  if e.isdigit():
   return True
  #this will allow backspace to work
  elif e=="":
   return True
  else:
   return False


 def comparealg(self):
   #self.df_classifier.set_index(['Algorithm'], inplace=True)
    fobj=plt.figure(figsize=(6,4),facecolor='#00FF00')
    spobj=fobj.add_subplot(1,1,1)
    alg = self.df_classifier['Algorithm'].tolist()
    accuracy1 = self.df_classifier['Accuracy'].tolist()
    
    x_val=np.arange(len(alg))
    spobj.bar(x_val,accuracy1)
    spobj.set_xticks(x_val)
    spobj.set_xticklabels(alg)
    spobj.set_xlabel('Algorithm')
    spobj.set_title('Algoritm Comaparision')
    #spobj.set_xticks(accuracy1)
    plt.show()


 def bestmodel(self):
     self.df3=self.df_classifier.nlargest(1, ['Accuracy'])
     
     print('Best Algoriyhm Model')
     print(self.df3)
     print('--------------------------')



 def modellingg(self):
    ### 1. Logistic Regression

# logistic regression again with the balanced dataset

  normalized_df_reg = LogisticRegression().fit(self.X_train, self.y_train)

  normalized_df_reg_pred =normalized_df_reg.predict(self.X_test)
  print('-------------LR-------------')
  print(confusion_matrix(self.y_test, normalized_df_reg_pred))

# check accuracy: Accuracy: Overall, how often is the classifier correct? Accuracy = (True Pos + True Negative)/total
  acc = accuracy_score(self.y_test, normalized_df_reg_pred)
  print(f"The accuracy score for LogReg is: {round(acc,3)*100}%")

# f1 score: The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.
  f1 = f1_score(self.y_test, normalized_df_reg_pred)
  print(f"The f1 score for LogReg is: {round(f1,3)*100}%")

# Precision score: When it predicts yes, how often is it correct? Precision=True Positive/predicted yes
  precision = precision_score(self.y_test, normalized_df_reg_pred)
  print(f"The precision score for LogReg is: {round(precision,3)*100}%")

# recall score: True Positive Rate(Sensitivity or Recall): When it’s actually yes, how often does it predict yes? True Positive Rate = True Positive/actual yes
  recall = recall_score(self.y_test, normalized_df_reg_pred)
  print(f"The recall score for LogReg is: {round(recall,3)*100}%")
  print('--------------------------')

### 2. SVM


# Support Vector Machine

#initialize model
  svm = SVC()

#fit model
  svm.fit(self.X_train, self.y_train)

  normalized_df_svm_pred = svm.predict(self.X_test)
  print('-------------svm-------------')
  print(confusion_matrix(self.y_test, normalized_df_svm_pred))

# check accuracy: Accuracy: Overall, how often is the classifier correct? Accuracy = (True Pos + True Negative)/total
  acc = accuracy_score(self.y_test, normalized_df_svm_pred)
  print(f"The accuracy score for SVM is: {round(acc,3)*100}%")

# f1 score: The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.
  f1 = f1_score(self.y_test, normalized_df_svm_pred)
  print(f"The f1 score for SVM is: {round(f1,3)*100}%")

# Precision score: When it predicts yes, how often is it correct? Precision=True Positive/predicted yes
  precision = precision_score(self.y_test, normalized_df_svm_pred)
  print(f"The precision score for SVM is: {round(precision,3)*100}%")

# recall score: True Positive Rate(Sensitivity or Recall): When it’s actually yes, how often does it predict yes? True Positive Rate = True Positive/actual yes
  recall = recall_score(self.y_test, normalized_df_svm_pred)
  print(f"The recall score for SVM is: {round(recall,3)*100}%")
  print('--------------------------')
  
### 3. Decision Tree

# Decision Tree

#initialize model
  self.dtc_up = DecisionTreeClassifier()

# fit model
  self.dtc_up.fit(self.X_train,self.y_train)

  normalized_df_dtc_pred = self.dtc_up.predict(self.X_test)
  print('-------------DT-------------')
  print(confusion_matrix(self.y_test, normalized_df_dtc_pred))

# check accuracy: Accuracy: Overall, how often is the classifier correct? Accuracy = (True Pos + True Negative)/total
  acc = accuracy_score(self.y_test, normalized_df_dtc_pred)
  print(f"The accuracy score for DTC is: {round(acc,3)*100}%")

# f1 score: The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.
  f1 = f1_score(self.y_test, normalized_df_dtc_pred)
  print(f"The f1 score for DTC is: {round(f1,3)*100}%")

# Precision score: When it predicts yes, how often is it correct? Precision=True Positive/predicted yes
  precision = precision_score(self.y_test, normalized_df_dtc_pred)
  print(f"The precision score for DTC is: {round(precision,3)*100}%")

# recall score: True Positive Rate(Sensitivity or Recall): When it’s actually yes, how often does it predict yes? True Positive Rate = True Positive/actual yes
  recall = recall_score(self.y_test, normalized_df_dtc_pred)
  print(f"The recall score for DTC is: {round(recall,3)*100}%")
  print('--------------------------')
  
### 4. KNN

# KNN Model

#initialize model
  self.knn = KNeighborsClassifier(n_neighbors = 2)

#fit model
  self.knn.fit(self.X_train, self.y_train)
  
# prediction = knn.predict(x_test)
  #print(self.X_test)
  normalized_df_knn_pred = self.knn.predict(self.X_test)
  #print(normalized_df_knn_pred)

  print('-------------KNN-------------')
  print(confusion_matrix(self.y_test, normalized_df_knn_pred))


# check accuracy: Accuracy: Overall, how often is the classifier correct? Accuracy = (True Pos + True Negative)/total
  acc = accuracy_score(self.y_test, normalized_df_knn_pred)
  print(f"The accuracy score for KNN is: {round(acc,3)*100}%")

# f1 score: The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.
  f1 = f1_score(self.y_test, normalized_df_knn_pred)
  print(f"The f1 score for KNN is: {round(f1,3)*100}%")

# Precision score: When it predicts yes, how often is it correct? Precision=True Positive/predicted yes
  precision = precision_score(self.y_test, normalized_df_knn_pred)
  print(f"The precision score for KNN is: {round(precision,3)*100}%")

# recall score: True Positive Rate(Sensitivity or Recall): When it’s actually yes, how often does it predict yes? True Positive Rate = True Positive/actual yes
  recall = recall_score(self.y_test, normalized_df_knn_pred)
  print(f"The recall score for KNN is: {round(recall,3)*100}%")
  print('--------------------------')

### 5. NaiveBayes

#initialize model
  gnb = GaussianNB()

# fit model
  gnb=gnb.fit(self.X_train, self.y_train)

  normalized_df_gnb_pred = gnb.predict(self.X_test)
  print('-------------NB-------------')
  print(confusion_matrix(self.y_test, normalized_df_gnb_pred))

# check accuracy: Accuracy: Overall, how often is the classifier correct? Accuracy = (True Pos + True Negative)/total
  acc = accuracy_score(self.y_test, normalized_df_gnb_pred)
  print(f"The accuracy score for NaiveBayes is: {round(acc,3)*100}%")

# f1 score: The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.
  f1 = f1_score(self.y_test, normalized_df_gnb_pred)
  print(f"The f1 score for NaiveBayes is: {round(f1,3)*100}%")

# Precision score: When it predicts yes, how often is it correct? Precision=True Positive/predicted yes
  precision = precision_score(self.y_test, normalized_df_gnb_pred)
  print(f"The precision score for NaiveBayes is: {round(precision,3)*100}%")

# recall score: True Positive Rate(Sensitivity or Recall): When it’s actually yes, how often does it predict yes? True Positive Rate = True Positive/actual yes
  recall = recall_score(self.y_test, normalized_df_gnb_pred)
  print(f"The recall score for NaiveBayes is: {round(recall,3)*100}%")
  print('--------------------------')



 def Pipemodel(self):
     
## Test - Train Split

  
# clarify what is y and what is x label
  self.new_features=self.df[['sysBP', 'glucose','age','totChol','cigsPerDay','diaBP','prevalentHyp','diabetes','BPMeds','male','TenYearCHD']]
  #x=self.new_features.iloc[:,:-1]
 # y=self.new_features.iloc[:,-1]
  y = self.new_features['TenYearCHD']
  X = self.new_features.drop(['TenYearCHD'], axis = 1)

# divide train test: 80 % - 20 %
  self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.2, random_state=29)



## **Model Pipeline**

  self.df_classifier = pd.DataFrame(columns = ['Algorithm', 'Accuracy']) 

  self.y_train = self.new_features['TenYearCHD']
  self.X_train = self.new_features.drop('TenYearCHD', axis=1)
  #print("hai")
  from sklearn.pipeline import Pipeline

  classifiers = [LogisticRegression(),SVC(),DecisionTreeClassifier(),KNeighborsClassifier(2),GaussianNB()]
  classifier_name=['LogisticRegression','SVM','DecisionTree','KNN','NavieBayes']
  i=0
  for classifier in classifiers:
    pipe = Pipeline(steps=[('classifier', classifier)])
    pipe.fit(self.X_train,self.y_train)   
   # print("The accuracy score of {0} is: {1:.2f}%".format(classifier,(pipe.score(X_test, y_test)*100)))
    self.df_classifier.loc[i] = [classifier_name[i],pipe.score(self.X_test, self.y_test)*100]
    i=i+1
  print(self.df_classifier)
  print('--------------------------')



 def feature_imp(self):

# Identify the features with the most importance for the outcome variable Heart Disease

  from sklearn.feature_selection import SelectKBest
  from sklearn.feature_selection import chi2

# separate independent & dependent variables
  X = self.df.iloc[:,0:14]  #independent columns
  y = self.df.iloc[:,-1]    #target column i.e price range

# apply SelectKBest class to extract top 10 best features
  bestfeatures = SelectKBest(score_func=chi2, k=10)
  fit = bestfeatures.fit(X,y)
  dfscores = pd.DataFrame(fit.scores_)
  dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 
  featureScores = pd.concat([dfcolumns,dfscores],axis=1)
  featureScores.columns = ['Specs','Score']  #naming the dataframe columns
  print('10 Best Features')
  print(featureScores.nlargest(11,'Score'))  #print 10 best features

  print("--------------------------------")
  
  featureScores = featureScores.sort_values(by='Score', ascending=False)
  print('All Feature Scores')
  print(featureScores)
  print("--------------------------------")

# visualizing feature selection
  plt.figure(figsize=(20,5))
  sns.barplot(x='Specs', y='Score', data=featureScores, palette = "GnBu_d")
  plt.box(False)
  plt.title('Feature importance', fontsize=16)
  plt.xlabel('\n Features', fontsize=14)
  plt.ylabel('Importance \n', fontsize=14)
  plt.xticks(fontsize=12)
  plt.yticks(fontsize=12)
  plt.show()


# selecting the 10 most impactful features for the target variable
  features_list = featureScores["Specs"].tolist()[:10]
  print('Feature List')
  print(features_list)
  print("--------------------------------")

# Create new dataframe with selected features

  self.df = self.df[['sysBP', 'glucose','age','totChol','cigsPerDay','diaBP','prevalentHyp','diabetes','BPMeds','male','TenYearCHD']]


# Zooming into cholesterin outliers

  #sns.boxplot(self.df.totChol)
  outliers = self.df[(self.df['totChol'] > 500)] 
 # print(outliers)

# Dropping 2 outliers in cholesterin
  self.df = self.df.drop(self.df[self.df.totChol > 599].index)
 # sns.boxplot(self.df.totChol)
  self.df_clean = self.df
  #print(self.df_clean.head(10))

  scaler = MinMaxScaler(feature_range=(0,1)) 

#assign scaler to column:
  self.df_scaled = pd.DataFrame(scaler.fit_transform(self.df_clean), columns=self.df_clean.columns)
 # print(self.df_scaled.head(10))







if __name__ == '__main__':

 root = Tk()

 application=HeartPrediction(root)
 #root.geometry('500x500') 
 root.mainloop()
