#lets learn decsion tree algorithm and random forest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,recall_score,precision_score
df=pd.read_csv('tested.csv')
#print(f.info())
df['Age']=df['Age'].fillna(df['Age'].median())
df['Fare']=df['Fare'].fillna(df['Fare'].median())
df.drop(['Name','Cabin','Ticket','PassengerId','Sex'],axis=1,inplace=True)
#df['Sex']=df['Sex'].map({'male':1,'female':0})
df = pd.get_dummies(df, columns=['Embarked'])
#print(df.dtypes)
#print(df.info())
#now its turn to split data
df['family_size']=df['Parch']+df['SibSp']+1
x=df.drop(['Survived','SibSp','Parch'],axis=1)
y=df['Survived']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=RandomForestClassifier(class_weight='balanced',n_estimators=100,random_state=42)
model.fit(x_train,y_train)
#lets predict
y_pred=model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

print(f"Accuracy:  {acc:.2%}")
print(f"Precision: {pre:.2%}")
print(f"Recall:    {rec:.2%}")