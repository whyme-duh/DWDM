import pandas as pd
from IPython.display import display_html
from sklearn import tree
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def toy_dataset():
    animal = [['human', 1, 1, 0, 0,1,0,'mammals'],['python',0,0,0,0,0,1,'reptiles'],
    ['salmon',0,0,1,0,0,0,'fishes'], ['whale',1,1,1,0,0,0, 'mammals'],
    ['frog', 0,0,1,0,1,1,'amphibians'],['komodo',0,0,0,0,1,0,'reptiles'],
    ['bat', 1,1,0,1,1,1,'mammals'],['pigeon',1,0,0,1,1,0, 'birds'],
    ['cat', 1,1,0,0,1,0, 'mammals'],['leopard shark',0,1,1,0,0,0, 'fishes'],
    ['turtle',0,0,1,0,1,0, 'reptiles'],['penguin', 1,0,1,0,1,0, 'birds'],
    ['porcupine', 1,1,0,0,1,1,'mammals'], ['ee!',0,0,1,0,0,0, 'fishes'],
    ['salamander',0,0,1,0,1,1, 'amphibians']]
    titles =['Name', 'Warm_blooded', 'Give_birth','Aquatic_creature', 'Aerial_reature','Has_legs', 'Hibernates', 'Class']
    data = pd.DataFrame(animal,columns=titles)
    data['Class'] = data['Class'].replace(['fishes', 'birds', 'amphibians', ' reptiles'], 'non-mammals')
    print ("Do you want to view data?")
    choice = input()
    if choice=='yes':
        display_html(data)
    return data

def build_model(data):
    Y = data['Class']
    X = data.drop(['Name', 'Class'],axis=1)
    clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=3)
    clf = clf.fit(X, Y)
    return clf

def prediction_using_model(clf):
    testData =[['gila monster',0,0,0,0,1,1, 'non-mammals'],
    ['platypus', 1,0,0,0,1,1,'mammals'],
    ['owl', 1,0,0,1,1,0,'non-mammals'],
    ['dolphin', 1,1,1,0,0,0, 'mammals']]
    titles = ['Name', 'Warm_blooded','Give_birth','Aquatic_creature','Aerial_reature', 'Has_legs', 'Hibernates', 'Class']
    testData = pd.DataFrame(testData, columns=titles)
    print("Do you want to view test data?")
    choice=input()
    if choice== 'yes':
        display_html(testData)
    #Splitting test data
    y_test = testData['Class']
    x_test = testData.drop(['Name', 'Class'],axis=1)
    y_pred = clf.predict(x_test)
    predictions = pd.concat([testData['Name'],pd.Series(y_pred, name='Predicted Class')], axis=1)
    print("Prediction for your test data is:")
    display_html(predictions)
    print("Do you want to view Evaluation of Model?")
    choice = input()
    if choice == 'yes':
        model_evaluation(y_pred, y_test)
    else:
        quit()

def model_evaluation(y_pred,y_test):
    print("Confusion Matrix:")
    report=(confusion_matrix(y_test, y_pred))
    cf=pd.DataFrame(report).transpose()
    display_html(cf)
    score = accuracy_score(y_test,y_pred)
    print('Decision Tree Accuracy :', score)
    print("Classification report:")
    report=(classification_report(y_test, y_pred, output_dict=True))
    df = pd.DataFrame(report).transpose()
    display_html(df[['precision', 'recall', 'f1-score']].head (2))

def main():
    data=toy_dataset()
    model= build_model(data)
    # to visualize tree install both graphviz and pydotplus
    dot_data = tree.export_graphviz(model, 'tree.dot',class_names=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png('tree.png')
    print ("Your decision tree constructed successfully, check the current directory for tree.png")
    prediction_using_model(model)

main()


