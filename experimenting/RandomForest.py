from dataset import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()
x,y = load_dataset()
# x = scaler.fit_transform(x)
print(x.shape,y.shape)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)


# Random forest model
rf_model = RandomForestClassifier()
rf_model.fit(x_train,y_train)
y_pred = rf_model.predict(x_test)

# model evaluation

print('Accuracy of Random Forest model',accuracy_score(y_test,y_pred)*100)
print('Confusion Matrix of Random Forest model\n',confusion_matrix(y_test,y_pred))
