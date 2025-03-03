import pandas as pd # helps organize patients data in tables like excel sheets
import numpy as np # helps with numbers and calculations
from sklearn.model_selection import train_test_split # split data into "study" and "test" groups
from sklearn.preprocessing import StandardScaler # makes all numbers even
from sklearn.linear_model import LogisticRegression # the ai that will learn
from sklearn.metrics import accuracy_score # checks how good the ai is
import pickle # saves the ai trained data for later use


data = {
    'fever': [101, 98, 104, 97, 103, 99, 105, 98, 100, 102],
    'headache': [1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
    'lack_of_appetite': [1, 0, 1, 0, 1, 1, 1, 0, 1, 1],
    'body_pains': [1, 0, 1, 0, 1, 1, 1, 0, 1, 1],
    'fatigue': [1, 0, 1, 0, 1, 1, 1, 0, 1, 1],
    'malaria': [1, 0, 1, 0, 1, 0, 1, 0, 1, 1]
}

df = pd.DataFrame(data)
print(df.head())

x = df[['fever', 'headache', 'lack_of_appetite', 'body_pains', 'fatigue']]
y = df['malaria']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = LogisticRegression() # creating the ai doctor
model.fit(x_train, y_train) # trains the ai doctor on past patient data

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy * 100:.2f}%')

with open('malaria.model.plk', 'wb') as f:
    pickle.dump(model, f) #saves the trained ai doctor

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f) #saves the number standardizer