import pandas as pd
# Loading Data from the link
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/delaney_solubility_with_descriptors.csv')
print(df)

# Data preparation && separation as x and y
y = df['logS']
# print(y)
x=df.drop('logS', axis= 1)
# print(x)

# Data splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
# Model building
# Linear regression
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
# Training the model
lr.fit(X_train, y_train)

#Applying the model to make predictions
y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)
# print(y_lr_train_pred,y_lr_test_pred)

# Evaluating model performance
from sklearn.metrics import mean_squared_error, r2_score

lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)
# print('LR MSE (Train) : ' , lr_train_mse )
# print('LR MSE (Train) : ' , lr_train_r2)
# print('LR MSE (Test) : ' , lr_test_mse)
# print('LR MSE (Test) : ' , lr_test_r2)

lr_results = pd.DataFrame(['Linear regression ', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method' , 'Training MSE' , 'Training R2' , 'Test MSE' , 'Test R2']
print(lr_results)

# Random forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(X_train, y_train)
y_rf_train_pred = rf.predict(X_train)
y_rf_test_pred = rf.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score

rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)

rf_results = pd.DataFrame(['Random Forest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_results.columns = ['Method' , 'Training MSE' , 'Training R2' , 'Test MSE' , 'Test R2']
print(rf_results)

# Model comparison
df_models = pd.concat([lr_results, rf_results], axis=0)
print(df_models)

df_models.reset_index(drop=True)


# Data visualization of prediction results
import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(5, 5))
plt.scatter(x=y_train, y=y_lr_train_pred, c="#7CAE00", alpha=0.3)

z = np.polyfit(y_train, y_lr_train_pred, 1)
p = np.poly1d(z)

plt.plot(y_train, p(y_train), '#F8766D')
plt.ylabel('Predicted LogS')
plt.xlabel('Experimental LogS')
plt.show()


