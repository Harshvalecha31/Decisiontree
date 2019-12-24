from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict,train_test_split

# fetch the data
data = load_iris()
#print(data.target_names) to get the target names
#print(data.target)- to get the target variables!!

features = data.data
target = data.target
# train test split of the data
x_train , x_test , y_train , y_test = train_test_split(features, target,test_size=0.3)

# creating the model
model = DecisionTreeClassifier(criterion = 'entropy')
#by default gini index is used in this case we are using shanon entropy and information gain
model.fit(x_train,y_train)

# make predictions!!!
prediction = model.predict(x_test)

# get the accuracy score and confusion matrix
print('Confusion Matrix:',confusion_matrix(y_test,prediction))
print('Accuracy score:',accuracy_score(y_test, prediction))

# applying K cross validations
estimation = cross_val_predict(model,features,target,cv = 10)
print('confusion matrix 2: ',confusion_matrix(target,estimation))
print('Accuracy Score 2:',accuracy_score(target,estimation))

# to get the decision path of the tree 
#print(model.decision_path(x_test))
# to get depth of the decision tree!!
#print(model.get_depth()) In this case the depth was 5
# to get the leave nodes count of the tree
#print(model.get_n_leaves()) In this case there were 7 leave nodes!!
