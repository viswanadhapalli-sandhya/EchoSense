from dataset import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x,y = load_dataset()
x = scaler.fit_transform(x)
print(x.shape,y.shape)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)


# radial svm
r_svm = SVC()
r_svm.fit(x_train,y_train)
r_pred = r_svm.predict(x_test)

# model evaluation

print('Accuracy of Radial SVM',accuracy_score(y_test,r_pred)*100)
print('Confusion Matrix of Radial SVM\n',confusion_matrix(y_test,r_pred))

# linear svm
l_svm = SVC(kernel="linear")
l_svm.fit(x_train,y_train)
l_pred = l_svm.predict(x_test)

# model evaluation
print('Accuracy of Linear SVM',accuracy_score(y_test,l_pred)*100)
print('Confusion Matrix of Linear SVM\n',confusion_matrix(y_test,l_pred))

# poly svm
p_svm = SVC(kernel="poly")
p_svm.fit(x_train,y_train)
p_pred = p_svm.predict(x_test)

# model evaluation
print('Accuracy of Poly SVM',accuracy_score(y_test,p_pred)*100)
print('Confusion Matrix of Poly SVM\n',confusion_matrix(y_test,p_pred))