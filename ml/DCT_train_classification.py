import sys
import pickle
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier


path = os.path.join(os.path.dirname(__file__),"..","log")
allFile = os.listdir(path)
data_set = []
for file in allFile[:]:
    with open(os.path.join(path,file),"rb") as f:
        data_set.append(pickle.load(f))
        
Ball_x = []
Ball_y = []
Vector_x = []
Vector_y = []
Direction = []
Platform = []
Command = []

for data in data_set:
    for i, sceneInfo in enumerate(data["scene_info"][2:-2]):
        Ball_x.append(sceneInfo['ball'][0])
        Ball_y.append(sceneInfo['ball'][1])
        Platform.append(sceneInfo['platform'][0])
        Vector_x.append(data['scene_info'][i+1]["ball"][0] - data['scene_info'][i]["ball"][0])
        Vector_y.append(data['scene_info'][i+1]["ball"][1] - data['scene_info'][i]["ball"][1])
        if Vector_x[-1] > 0:
            if Vector_y[-1] > 0:
                Direction.append(0)
            else:
                Direction.append(1)
        else:
            if Vector_y[-1] > 0:
                Direction.append(2)
            else:
                Direction.append(3)
    for command in data['command'][2:-2]:
        if command == "NONE" or command == "":
            Command.append(0)
        elif command == "MOVE_LEFT":
            Command.append(-1)
        elif command == "MOVE_RIGHT":
            Command.append(1)
        else:
            Ball_x = Ball_x[:-1]
            Ball_y = Ball_y[:-1]
            Platform = Platform[:-1]
            Vector_x = Vector_x[:-1]
            Vector_y = Vector_y[:-1]
            Direction = Direction[:-1]

Command = np.array(Command)
Command = Command.reshape(len(Command),1)

X = np.array([0,0,0,0,0,0])
for i in range(len(Ball_x)):
    X = np.vstack((X, [Ball_x[i], Ball_y[i], Vector_x[i], Vector_y[i], Direction[i], Platform[i]]))
X = X[1::]
Y = Command[:,0]

#DecisionTree

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

d_value = sys.argv[1]

model = DecisionTreeClassifier(max_depth = d_value)
model = model.fit(x_train, y_train)


# y_predict = model.predict(x_test)
# Accuracy = float('{:.3f}'.format(accuracy_score(y_predict,y_test)))
# depth = model.tree_.max_depth
# print("depth = ", depth, "Accuracy = ", Accuracy)

# F1Score = float('{:.3f}'.format(f1_score(y_test, y_predict, average="weighted")))
# print("F1 score = ", F1Score)

# save model

path = os.path.join(os.path.dirname(__file__), "save")
if not os.path.isdir(path):
    os.mkdir(path)

with open(os.path.join(os.path.dirname(__file__),'save',\
    "DCT_classification.pickle"),'wb') as f:
    pickle.dump(model,f)

    