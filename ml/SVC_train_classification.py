import pickle
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC


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
Command = Command.reshape(len(Command), 1)

X = np.array([0,0,0,0,0,0])
for i in range(len(Ball_x)):
    X = np.vstack((X, [Ball_x[i], Ball_y[i], Vector_x[i], Vector_y[i], Direction[i], Platform[i]]))
X = X[1::]
Y = Command[:,0]

#SVM
print("Starting program...(1)")
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)


#SVM 參數調整
C_range = [0.1, 1]
kernels = ['linear', 'rbf', 'poly']
best_C = 0
best_kernel = ''
best_accuracy = 0
best_f1_score = 0

for C in C_range:
    for kernel in kernels:
        model = SVC(C=C, kernel=kernel)
        model.fit(x_train, y_train)
        y_predict = model.predict(x_val)
        
        acc = accuracy_score(y_val, y_predict)
        f1 = f1_score(y_val, y_predict, average='weighted')

        print(f"C = {C}, Kernel = {kernel}, Accuracy = {acc:.3f}, F1 Score = {f1:.3f}")

        if acc > best_accuracy:
            best_accuracy = acc
            best_f1_score = f1
            best_C = C
            best_kernel = kernel

print("Starting program...(2)")
model = SVC(C = best_C, kernel = best_kernel)
model = model.fit(x_train, y_train)
y_predict = model.predict(x_test)
print("Starting program...(3)")

Accuracy = float('{:.3f}'.format(accuracy_score(y_predict, y_test)))
F1Score = float('{:.3f}'.format(f1_score(y_test, y_predict, average='weighted')))

print(f"Best C = {best_C}, Best Kernel = {best_kernel}")
print(f"Testing Accuracy = {Accuracy}, Testing F1 Score = {F1Score}")

# save model

path = os.path.join(os.path.dirname(__file__), "save")
if not os.path.isdir(path):
    os.mkdir(path)

# with open(os.path.join(os.path.dirname(__file__),'save',\
#     "SVM_classification_acc={:.2f}_data={}.pickle".format( Accuracy, len(X))),'wb') as f:
#     pickle.dump(model,f)
    
with open(os.path.join(os.path.dirname(__file__),'save',\
    "SVM_classification_C={}_karnel={}_acc={:.2f}_data={}.pickle".format(best_C, best_kernel, Accuracy, len(X))),'wb') as f:
    pickle.dump(model,f)

    