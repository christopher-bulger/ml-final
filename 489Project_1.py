import numpy as np
import pandas as pd
from collections import Counter

#Calculates the euclidean disatnce of an n-dimensional data set.
def eucDist(p1, p2):
    sum = 0
    for i in range(1, len(p1)):
            sum += ((p1[i] - p2[i]) ** 2)
    return np.sqrt(sum)

Results = pd.read_csv('2012-18_teamBoxScore.csv')
trainingData = Results[['teamAbbr','teamLoc', 'teamRslt', 'teamPTS', 'opptPTS']]
trainingData = np.array(trainingData)

# Converting teamRslt and teamLoc to binary
for i in range(0, len(trainingData)):
    if trainingData[i][1] == 'Home':
        trainingData[i][1] = 1
    else:
        trainingData[i][1] = 0
    if trainingData[i][2] == 'Win':
        trainingData[i][2] = 1
    else:
        trainingData[i][2] = 0

# Calculating win/loss margin
for i in range(0, len(trainingData)):
    trainingData[i][3] = trainingData[i][3] / trainingData[i][4]

trainingData = np.delete(trainingData, 4, 1)
testData = np.delete(trainingData, slice(0, 13386), 0)
trainingData = np.delete(trainingData, slice(13386, 14757), 0)
trainingData = np.delete(trainingData, 13386, 0)

K_val = 30
#K_val = int(np.sqrt(len(trainingData)))
correctly_classified = 0

for i in range (0, len(testData)):
    distances = list()
    k_nearest_labels = list()

    for j in range (0, len(trainingData)):
        distances.append(eucDist(testData[i], trainingData[j])) # List of distances

    sorted_by_index = np.argsort(distances)

    for j in range(0, K_val):
        k_nearest_labels.append(trainingData[sorted_by_index[j]][0])

    predicted_label = Counter(k_nearest_labels).most_common(1)[0][0]

    if predicted_label == testData[i][0]:
        correctly_classified += 1

accuracy = (correctly_classified / len(testData)) * 100
print("Accuracy is " + str(accuracy) + "%.")

