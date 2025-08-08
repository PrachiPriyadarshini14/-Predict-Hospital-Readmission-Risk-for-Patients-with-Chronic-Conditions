import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

df=pd.read_csv('healthcare_dataset.csv')
test_results_mapping = {'Normal': 0, 'Inconclusive': 2, 'Abnormal': 1}
df['Test_Results_Encoded'] = df['Test Results'].map(test_results_mapping)
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
df['Length of Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days
df['Name'] = df['Name'].str.lower()
name_count=df['Name'].value_counts()
df['frequency_of_times_admitted'] = df['Name'].map(name_count)
df['Readmission'] = ((df['frequency_of_times_admitted']>1) & (df['Test_Results_Encoded'] == 1) & (df['Length of Stay'] > 15)).astype(int)


x=df.drop(['Name','Gender','Blood Type','Medical Condition','Date of Admission','Doctor','Hospital','Insurance Provider','Billing Amount','Room Number','Admission Type','Discharge Date','Medication','Test Results','Readmission','Age'],axis=1)
knn=KNeighborsClassifier()
y=df['Readmission']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=2)
knn.fit(x_train,y_train)
pred=knn.predict(x_test)
score=accuracy_score(y_test,pred)

with open('model.pkl', 'wb') as f:
    pickle.dump(knn, f)

print("Model saved as 'model.pkl'")