from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import pandas as pd

columns=["age","menopause","tumor-size","inv-nodes","node-caps","deg-malig","breast","breast-quad","irradiat","Class"]

df=pd.read_csv('breast_cancer.csv',header=None,names=columns)

#feature types
ordinal_feat=["age","tumor-size","inv-nodes","deg-malig"]
nominal_feat=["menopause","node-caps","breast","breast-quad","irradiat"]
target="Class"

#encoding
#target column
label_encoder=LabelEncoder()
df[target]=label_encoder.fit_transform(df[target])

#ordinal encoding for ordered categorical variables
ordinal_encoder=OrdinalEncoder()
encoded_ordinal=ordinal_encoder.fit_transform(df[ordinal_feat])

#one-hot encoding for nominal categorical variables
onehot_encoder=OneHotEncoder(sparse_output=False,handle_unknown="ignore")
encoded_nominal=onehot_encoder.fit_transform(df[nominal_feat])

# Convert Encoded data to DataFrame
encoded_nominal_df = pd.DataFrame(encoded_nominal, columns=onehot_encoder.get_feature_names_out(nominal_feat))
encoded_ordinal_df = pd.DataFrame(encoded_ordinal, columns=ordinal_feat)
#merging the encoded data
df=df.drop(columns=nominal_feat+ordinal_feat)
df=pd.concat([df,encoded_nominal_df,encoded_ordinal_df],axis=1)

#now splitting
X=df.drop(columns=['Class'])
y=df['Class']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy:.4f}")

# Display the coefficients (theta values)
print("Theta (Coefficient) values for Logistic Regression:")
print(model.coef_)

