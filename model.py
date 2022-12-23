from google.colab import files
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
from graphviz import render
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# upload dataset
uploaded = files.upload()
iris = pd.read_csv('Iris.csv')
iris.drop('Id',axis=1,inplace=True)

X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm' ]]
y = iris['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)
tree_model = DecisionTreeClassifier() 
tree_model = tree_model.fit(X_train, y_train)

y_pred = tree_model.predict(X_test)
acc_secore = round(accuracy_score(y_pred, y_test), 3)
#print('Accuracy: ', acc_secore)

# predict the label of a new set of data
#print(tree_model.predict([[6.2, 3.4, 5.4, 2.3]])[0])    # ([[SepalLength, SepalWidth, PetalLength, PetalWidth]])

# the visualization of the decision tree
export_graphviz(
    tree_model,
    out_file = "iris_tree.dot",
    feature_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'],
    class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica' ],
    rounded= True,
    filled =True
)

render('dot', 'png', 'iris_tree.dot')
image_path = "/content/iris_tree.dot.png"
image = mpimg.imread(image_path)
plt.figure(figsize=(1000/float(80), 1000/float(80)), dpi=80)
plt.imshow(image)
plt.show()
