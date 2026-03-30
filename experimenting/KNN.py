from dataset import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x,y = load_dataset()
x = scaler.fit_transform(x)
print(x.shape,y.shape)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)


# knn model with 5 neighbors
knn_model = KNeighborsClassifier()
knn_model.fit(x_train,y_train)
y_pred = knn_model.predict(x_test)

# model evaluation

print('Accuracy of KNN model',accuracy_score(y_test,y_pred)*100)
print('Confusion Matrix of KNN model\n',confusion_matrix(y_test,y_pred))
