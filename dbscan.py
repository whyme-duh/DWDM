import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def toy_dataset():
    value = np.array([[1, 3], [4, 2], [2, 3],[6, 71], [8, 9], [10,12],[10,15],
                      [50, 19], [20, 20], [44, 21],[46, 18], [51, 19],
                      [50, 101], [40, 22], [50, 23], [49, 53], [50, 25],
                      [50, 50], [50, 45], [44, 51],[46, 55], [51, 48],
                      [50, 58], [40, 49], [50, 55],[49, 53], [50, 52],
                      [25,80], [100, 30],[150, 90]])
    titles = ['x','y']
    data = pd.DataFrame(value, columns=titles)
    print("First five data points:")
    print(data.head())
    print("Do you want to view scatter plot of data? (yes/no)")
    option = input()
    if option == 'yes':
        print("Data points scatter plot:")
        plt.scatter(data['x'], data['y'])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Scatter plot of data points')
        plt.show()
    return data

def Dbscan_clustering(data):
    db = DBSCAN(eps=10.5, min_samples=4).fit(data)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = pd.DataFrame(db.labels_, columns=['Cluster ID'])
    result = pd.concat((data, labels), axis=1)
    print(result)
    print("Clusters constructed by DBSCAN:")
    plt.scatter(result['x'], result['y'], c=result['Cluster ID'], cmap='jet')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Clusters constructed by DBSCAN')
    plt.colorbar(label='Cluster ID')
    plt.show()

def main():
    data = toy_dataset()
    Dbscan_clustering(data)

main()
