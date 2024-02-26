import pandas as pd
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

def toy_dataset():
    animal = [['human', 1, 1, 0, 0,1,0,'mammals'], ['python',0,0,0,0,0,1, 'reptiles'],
    ['salmon',0,0,1,0,0,0,'fishes'], ['whale', 1,1,1,0,0,0, 'mammals'],
    ['frog',0,0,1,0,1,1, 'amphibians'], ['komodo',0,0,0,0,1,0, 'reptiles'],
    ['bat', 1,1,0,1,1,1, 'mammals'],['pigeon', 1,0,0,1,1,0, 'birds'],
    ['cat', 1,1,0,0,1,0, 'mammals'],['leopard shark',0,1,1,0,0,0, 'fishes'],
    ['turtle',0,0,1,0,1,0,'reptiles'] ,['penguin', 1,0,1,0,1,0, 'birds'],
    ['porcupine', 1,1,0,0,1,1,'mammals'], ['eel',0,0,1,0,0,0, 'fishes'],
    ['salamander',0,0,1,0,1,1, 'amphibians']]
    titles= ['Name', 'Warm_blooded', 'Give_birth', 'Aquatic_creature', 'Aerial_reature', 'Has_legs', 'Hibernates', 'Class']
    data = pd.DataFrame(animal,columns= titles)
    print("View data:")
    print(data)
    return data

def ward(names,X, Y):
    Z = hierarchy.linkage(X.values, 'ward')
    dn = hierarchy.dendrogram(Z, labels=names.tolist(), orientation='right')
    plt.title('Ward Hierarchical Clustering Dendrogram')
    plt.xlabel('Distance')
    plt.ylabel('Sample index')
    plt.show()

def centroid(names,X, Y):
    Z = hierarchy.linkage(X.values, 'centroid')
    dn = hierarchy.dendrogram(Z, labels=names.tolist(), orientation='right')
    plt.title('Centroid Hierarchical Clustering Dendrogram')
    plt.xlabel('Distance')
    plt.ylabel('Sample index')
    plt.show()

def group_average(names,X, Y):
    Z = hierarchy.linkage(X.values, 'average')
    dn = hierarchy.dendrogram(Z, labels=names.tolist(), orientation='right')
    plt.title('Group Average Hierarchical Clustering Dendrogram')
    plt.xlabel('Distance')
    plt.ylabel('Sample index')
    plt.show()

def complete_link(names,X, Y):
    Z = hierarchy.linkage(X.values, 'complete')
    dn = hierarchy.dendrogram(Z, labels=names.tolist(), orientation='right')
    plt.title('Complete Linkage Hierarchical Clustering Dendrogram')
    plt.xlabel('Distance')
    plt.ylabel('Sample index')
    plt.show()

def single_link(names, X, Y):
    Z = hierarchy.linkage(X.values, 'single')
    dn = hierarchy.dendrogram(Z, labels=names.tolist(), orientation='right')
    plt.title('Single Linkage Hierarchical Clustering Dendrogram')
    plt.xlabel('Distance')
    plt.ylabel('Sample index')
    plt.show()

def main():
    data = toy_dataset()
    names = data['Name']
    Y = data['Class']
    X = data.drop(['Name','Class'], axis=1)
    print("Your data is ready!")
    print("Select your option:")
    print("1. Single Linkage")
    print("2. Complete Linkage")
    print("3. Group Average")
    print("4. Centroid")
    print("5. Ward")
    choice = int(input())
    if choice == 1:
        single_link(names, X, Y)
    elif choice == 2:
        complete_link(names, X, Y)
    elif choice == 3:
        group_average(names, X, Y)
    elif choice == 4:
        centroid(names, X, Y)
    elif choice == 5:
        ward(names, X, Y)
    else:
        print("Enter correct choice next time")
        quit()

main()
