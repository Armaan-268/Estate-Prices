# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
    # Importing the dataset
    dataset = pd.read_csv('dataset.csv')
    x = dataset.iloc[1:, 2:-1].values
    y = dataset.iloc[1:, -1].values
    y = y.reshape(len(y),1)
    t = dataset.iloc[364:, 2:3].values
    
    #Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    x = sc_x.fit_transform(x)
    y = sc_y.fit_transform(y)
    
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.12, random_state = 0,shuffle = 'True')

    # Training the Regression model on the Training set
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)

    # Training the Polynomial Regression model on the Training dataset
    from sklearn.preprocessing import PolynomialFeatures
    poly_reg = PolynomialFeatures(degree = 3)
    x_poly = poly_reg.fit_transform(x_train)
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(x_poly, y_train)

    # Predicting the Test set results
    y_pred = sc_y.inverse_transform(lin_reg_2.predict(poly_reg.fit_transform(x_test)))
    y_test = sc_y.inverse_transform(y_test)
    m1 = max(y_pred)
    m2 = max(y_test)
    #print(y_pred)

    #Ploting Graph : Plot is b/w Y_test set and predicted Y set
    z = [0,m1,m2]   
    plt.plot(z,z,color = 'b',label = 'X=Y line')  
    plt.scatter(y_test,y_pred,color = 'r',label = 'Y_test,Y_pred')
    plt.title('Y_test vs Y_Predict')
    plt.xlabel('Y_test')
    plt.ylabel('Y_pred')
    plt.show()

    #Ploting Graph 2: Age of house vs Price/area
    plt.scatter(t,y_test,color = 'b')
    plt.scatter(t,y_pred,color = 'r')
    plt.title('Age vs Price')
    plt.xlabel('Age of House')
    plt.ylabel('Price per Area')
    plt.show()
    
    #Finding Root Mean Square Error
    def rmse(predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())
    print('RMSE:',rmse(y_pred,y_test))

if __name__ == "__main__":
    main()