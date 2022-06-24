import pandas as pd
import plotly.express as px

df = pd.read_csv("Admission_Predict.csv")

toefl_score = df["TOEFL Score"].tolist()
gre_score = df["GRE Score"].tolist()

fig = px.scatter(x=toefl_score, y=gre_score)
fig.show()

import plotly.graph_objects as go

toefl_score = df["TOEFL Score"].tolist()
gre_score = df["GRE Score"].tolist()

chances_of_admit = df["Chance of admit"].tolist()
colors = []
for data in chances_of_admit:
    if data == 1:
        colors.append("green")
    else:
        colors.append("red")

fig = go.Figure(data = go.Scatter(
    x = toefl_score,
    y = gre_score,
    mode = 'markers',
    marker = dict(color = colors)
))
fig.show()  

score = df[["Chance of admit"]]

from sklearn.model_selection import train_test_spilt
score_train, score_test, chances_of_admit_train, chances_of_admit_test = train_test_spilt(score, chances_of_admit, test_size = 0.25, random_state = 0)
print(score_train)

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)
classifier.fil(score_train, chances_of_admit_test)

chances_of_admit_pred = classifier.predict(score_test)

from sklearn.metrics import accuracy_score
print ("Accuracy : ", accuracy_score(chances_of_admit_test, chances_of_admit_pred))


