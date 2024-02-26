import pandas as pd
from IPython.display import display_html
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def build_model(X_train, Y_train):
    clf = LinearSVC()
    clf = clf.fit(X_train, Y_train)
    return clf


def prediction_using_model(clf,X_test, Y_test):
    X_test = X_test.reset_index()
    del X_test['index']
    Y_test = Y_test.reset_index()
    del Y_test ['index']
    Y_pred = clf.predict(X_test)
    predictions = pd.concat([X_test,pd.Series(Y_pred,name='Predicted Class')], axis=1)
    print("Do you want to view the class label prediction for top five tuples of test data?")
    choice=input()
    if choice=='yes':
        display_html(predictions.head())
    print ("Do you want to view Evaluation of model?")
    choice=input()
    if choice=='yes':
        model_evaluation(Y_pred,Y_test)
    else:
        quit()

def model_evaluation(y_pred,y_test):
    print("Confusion Matrix:")
    report=(confusion_matrix(y_test, y_pred))
    cf=pd.DataFrame(report).transpose()
    display_html(cf)
    score = accuracy_score(y_test,y_pred)
    print('SVM Accuracy :',score)
    print("Classification report:")
    report=(classification_report(y_test, y_pred, output_dict=True))
    df = pd.DataFrame(report).transpose()
    display_html(df[['precision', 'recall','f1-score']].head(3))


def main():
    data = pd.read_csv('Iris.csv')
    print("Do you want to view to five data tuples of Iris Datasets?")
    choice = input()
    if choice =='yes':
        display_html(data.head())
    Y = data['Species']
    X = data.drop(['Id', 'Species'], axis = 1)
    X_train, X_test,Y_train, Y_test = train_test_split(X,Y,test_size = 0.25, random_state = 1)
    clf = build_model(X_train, Y_train)
    prediction_using_model(clf,X_test, Y_test)

main()