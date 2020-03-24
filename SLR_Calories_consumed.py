#Simple Linear Regression
#prediction of Calories consumed based on weight gained

# importing necessary libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

Calories_consumed=pd.read_csv("calories_consumed.csv")
Calories_consumed.columns

Calories_consumed.columns = ['Weight_gained', 'Calories_consumed'] 
Calories_consumed.columns 
 
plt.hist(Calories_consumed.Weight_gained)
plt.hist(Calories_consumed.Calories_consumed)

X=Calories_consumed.Weight_gained #weight gained
Y=Calories_consumed.Calories_consumed # all the columns in index 1

plt.plot(X,Y,"bo");plt.xlabel("Weight gained in grams");plt.ylabel("Calories Consumed")
X.corr(Y)
np.corrcoef(X,Y)


############################### Implementing the Linear Regression model from sklearn library
from sklearn.linear_model import LinearRegression
plt.scatter(X,Y,color='red')
regressor1=LinearRegression()
regressor1.fit(X.values.reshape(-1,1),Y)

#prediction of Y
y_pred=regressor1.predict(X.values.reshape(-1,1))

# Adjusted R-Squared value
regressor1.score(X.values.reshape(-1,1),Y) #0.897
rmse1 = np.sqrt(np.mean((y_pred-Y)**2)) # 232.833
regressor1.coef_
regressor1.intercept_

#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter(y_pred,(y_pred-Y),c="r")
plt.hlines(y=0,xmin=0,xmax=4000) 
# checking normal distribution for residual
plt.hist(y_pred-Y)

### Fitting Quadratic Regression 
Calories_consumed["Weight_Sq"] = Calories_consumed.Weight_gained*Calories_consumed.Weight_gained
regressor2 = LinearRegression()
regressor2.fit(X = Calories_consumed.iloc[:,[0,2]],y=Y)
y_pred2 = regressor2.predict(Calories_consumed.iloc[:,[0,2]])
# Adjusted R-Squared value
regressor2.score(Calories_consumed.iloc[:,[0,2]],Y)# 0.9078
rmse2 = np.sqrt(np.mean((y_pred2-Y)**2)) # 220.04
rmse2  
regressor2.coef_
regressor2.intercept_
#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter(y_pred2,(y_pred2-Y),c="r")
plt.hlines(y=0,xmin=0,xmax=4000)  
# Checking normal distribution
plt.hist(y_pred2-Y)
import pylab
import scipy.stats as st
st.probplot(y_pred2-Y,dist="norm",plot=pylab)

# Let us prepare a model by applying transformation on dependent variable
Calories_consumed["Calories_sqrt"] = np.sqrt(Calories_consumed.Calories_consumed)
regressor3 = LinearRegression()
regressor3.fit(X = Calories_consumed.iloc[:,[0,2]],y=Calories_consumed.Calories_sqrt)
y_pred3 = regressor3.predict(Calories_consumed.iloc[:,[0,2]])
# Adjusted R-Squared value
regressor3.score(Calories_consumed.iloc[:,[0,2]],Calories_consumed.Calories_sqrt)# 0.88
rmse3 = np.sqrt(np.mean(((y_pred3)**2-Y)**2)) # 227.571
rmse3
regressor3.coef_
regressor3.intercept_
#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter((y_pred3)**2,((y_pred3)**2-Y),c="r")
plt.hlines(y=0,xmin=0,xmax=4000)  
# checking normal distribution for residuals 
plt.hist(((y_pred3)**2-Y))
st.probplot((y_pred3)**2-Y,dist="norm",plot=pylab)




