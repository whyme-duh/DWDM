import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display_html
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def visualize(Ir, X, Y):
    plt.scatter(X, Y, color = "red")
    plt. plot(X, Ir.predict(X), color = "green")
    plt. title("Salary vs Experience (Testing Set)")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.show()

def linear(x_train, y_train):
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    m= lr.coef_
    b=lr.intercept_
    print("Learned parameter: \n slope=",m, "\nintercept=", b)
    return lr

def test(lr, x_test,y_test):
    x_test = x_test.reset_index()
    del x_test['index']
    y_test = y_test.reset_index()
    del y_test['index']
    y_pred = lr.predict(x_test)
    predictions = pd.concat([x_test,pd.Series(y_pred,name='Predicted salary')], axis=1)
    print("Do you want to view salary prediction of test data?")
    choice=input()
    if choice=='yes':
        display_html(predictions)
    print("Do you want to view Evaluation of linear regression model?")
    choice=input()
    if choice== 'yes':
        evaluation(lr,y_pred,y_test)
    else:
        quit()


def evaluation (lr,y_pred,y_test):
    print('Mean Absolute Error of the Model:',metrics.mean_absolute_error(y_test,y_pred))
    print('Mean Squared Error of the Model: ', metrics.mean_squared_error(y_test,y_pred))
    print('Root Mean Squared Error of the Model:', np.sqrt(metrics.mean_absolute_error(y_test,y_pred)))

def main():
    dataset = pd.read_csv('salary_Data.csv')
    print("Do you want to view the dataset?")
    choice=input
    if choice== 'yes':
        display_html(dataset)
    y = dataset ['Salary']
    x= dataset.drop(['Salary'],axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3)
    lr=linear(x_train,y_train)
    test(lr, x_test,y_test)
    print("Enter the year of experience to predict the salary of employee:")
    year=int(input())
    print("Salary for a employee with year of experience is:", lr.predict([[year]]))
    print("Do your want to view the ploat for test set prediction?")
    choice=input()
    if choice=='yes':
        print("Plotting the y test data vs y predicted data")
        visualize(lr, x_test,y_test)
    else:
        quit()

main()