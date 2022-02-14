import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn import preprocessing
import numpy.random as nr
from sklearn import model_selection as ms
import warnings
warnings.filterwarnings("ignore")
import classifires as clf
path = 'D:/Imarticus learning/projects_in class/ML_classifires/input/'

og_df = pd.read_csv(path+'German_Credit_Preped.csv')

og_df.head()

og_df = og_df.drop(['customer_id','foreign_worker','telephone'], axis=1)

for col in og_df.columns:
  if col == "bad_credit":continue
  if og_df[col].dtype != object:
    skew_value = og_df[col].skew()
    if skew_value>0.8:
        print(col, skew_value)
        og_df[col] = np.log2(og_df[col])
    
        
   #-----Lable encoding single column features-----
#print(og_df['purpose'].unique())
Features = og_df['credit_history']
enc = preprocessing.LabelEncoder()
enc.fit(Features)
Features = enc.transform(Features)
print(Features.shape)
ohe = preprocessing.OneHotEncoder()
encoded = ohe.fit(Features.reshape(-1,1))
Features = ohe.transform(Features.reshape(-1,1)).toarray()
print(Features.shape)
#print(oheFeatures[:10,:])

def encode_string(cat_feature):
    ## First encode the strings to numeric categories
    enc = preprocessing.LabelEncoder()
    enc.fit(cat_feature)
    enc_cat_feature = enc.transform(cat_feature)
    ## Now, apply one hot encoding
    ohe = preprocessing.OneHotEncoder()
    encoded = ohe.fit(enc_cat_feature.reshape(-1,1))
    return encoded.transform(enc_cat_feature.reshape(-1,1)).toarray()
    

categorical_columns = ['checking_account_status','purpose','savings_account_balance','time_employed_yrs','gender_status','other_signators','property','other_credit_outstanding','home_ownership','job_category']

for col in categorical_columns:
    temp = encode_string(og_df[col])
    Features = np.concatenate([Features, temp], axis = 1)

#print(Features.shape)
#print(Features[:2, :])    
  
Features = np.concatenate([Features,np.array(og_df[['loan_duration_mo','loan_amount','age_yrs','number_loans','dependents']])],axis=1)
#print(Features.shape)
#print(Features[:2, :])

nr.seed(9988)
labels = np.array(og_df['bad_credit'])
indx = range(Features.shape[0])
indx = ms.train_test_split(indx, test_size = 30)
x_train = Features[indx[0],:]
y_train = np.ravel(labels[indx[0]]) 
x_test = Features[indx[1],:]
y_test = np.ravel(labels[indx[1]])
print(x_train[:5,:] )

from sklearn.preprocessing import StandardScaler
scaler = preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train[:,:])
x_test = scaler.transform(x_test[:,:])
print(x_train.shape)
print(x_train[:5,:])

#dt = DecisionTreeClassifier(criterion='entropy',random_state = 112,class_weight = {1:0.5,0:0.5},max_depth = 4) 
dt = RandomForestClassifier(random_state=111,class_weight = {1:0.6,0:0.4},max_depth = 4,max_leaf_nodes=4,min_samples_split=4)
#dt = AdaBoostClassifier(random_state=111,learning_rate=0.5)
#dt = GradientBoostingClassifier(random_state=111, learning_rate=0.9, max_depth=3, max_leaf_nodes=4)
dt.fit(x_train,y_train)
test_pred = dt.predict(x_test)
train_pred = dt.predict(x_train)

from sklearn.metrics import classification_report
print(classification_report(y_test,test_pred))
print(classification_report(y_train,train_pred))
'''Y = og_df['bad_credit']
X = og_df.drop(['bad_credit'],axis=1)
print(X.info())

from sklearn.model_selection import train_test_split
trainX,trainY,testX,testY = train_test_split(X,Y)

des_tree = DecisionTreeClassifier(criterion='entropy')
d
es_tree.fit(trainX,trainY)'''