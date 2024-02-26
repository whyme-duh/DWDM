import pandas as pd
from sklearn.preprocessing import LabelEncoder

animal = [['human', 1, 1, 0, 0, 1, 0, 'mammals'], ['python', 0, 0, 0, 0, 0, 1, 'reptiles'],
          ['salmon', 0, 0, 1, 0, 0, 0, 'fishes'], ['whale', 1, 1, 1, 0, 0, 0, 'mammals'],
          ['frog', 0, 0, 1, 0, 1, 1, 'amphibians'], ['komodo', 0, 0, 0, 0, 1, 0, 'reptiles'],
          ['bat', 1, 1, 0, 1, 1, 1, 'mammals'], ['pigeon', 1, 0, 0, 1, 1, 0, 'birds'],
          ['cat', 1, 1, 0, 0, 1, 0, 'mammals'], ['leopard shark', 0, 1, 1, 0, 0, 0, 'fishes'],
          ['turtle', 0, 0, 1, 0, 1, 0, 'reptiles'], ['penguin', 1, 0, 1, 0, 1, 0, 'birds'],
          ['porcupine', 1, 1, 0, 0, 1, 1, 'mammals'], ['eel', 0, 0, 1, 0, 0, 0, 'fishes'],
          ['salamander', 0, 0, 1, 0, 1, 1, 'amphibians']]
titles = ['Name', 'Warm_blooded', 'Give_birth', 'Aquatic_creature', 'Aerial_creature', 'Has_legs', 'Hibernates', 'Class']

df = pd.DataFrame(animal, columns=titles)
print("Original DataFrame:")
print(df)

label_encoder = LabelEncoder()
df['Class'] = label_encoder.fit_transform(df['Class'])
print("\nDataFrame with Encoded 'Class' Column:")
print(df)

