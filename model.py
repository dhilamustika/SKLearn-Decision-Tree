# upload the dataset
from google.colab import files
uploaded = files.upload()

import pandas as pd
from sklearn.datasets import load_iris
 
# read the iris.csv file
iris = pd.read_csv('Iris.csv')

# display the first five rows of the dataframe by default
iris.head()

# print information about the dataframe
iris.info()

# remove unnecessary column
iris.drop('Id',axis=1,inplace=True)

# divide our dataset into features (X) and labels (y)
X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm' ]]
y = iris['Species']

# split the dataset into train and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)

from sklearn.tree import DecisionTreeClassifier
 
# create a Decision Tree model
tree_model = DecisionTreeClassifier() 
 
# train the model
tree_model = tree_model.fit(X_train, y_train)

# test the trained model
from sklearn.metrics import accuracy_score

y_pred = tree_model.predict(X_test)

acc_secore = round(accuracy_score(y_pred, y_test), 3)

print('Accuracy: ', acc_secore)

# predict the label of a new set of data using tree_model.predict([[SepalLength, SepalWidth, PetalLength, PetalWidth]])
print(tree_model.predict([[6.2, 3.4, 5.4, 2.3]])[0])

# the visualization of the decision tree
from sklearn.tree import export_graphviz
export_graphviz(
    tree_model,
    out_file = "iris_tree.dot",
    feature_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'],
    class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica' ],
    rounded= True,
    filled =True
)

# convert a .dot file to .png
from graphviz import render
render('dot', 'png', 'iris_tree.dot')

# display the visualization of the decision tree
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
image_path = "/content/iris_tree.dot.png"
image = mpimg.imread(image_path)
plt.figure(figsize=(1000/float(80), 1000/float(80)), dpi=80)
plt.imshow(image)
plt.show()
