import numpy
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler

def main():
    # Create minmax scaler
    scaler = MinMaxScaler()
    # Load team data into data frame
    standings = pd.read_csv('2012-18_standings.csv')
    y = pd.read_csv('2012-18_standings.csv').iloc[:, 2].to_numpy()
    # X = pd.read_csv('2012-18_standings.csv').iloc[:, 2].to_numpy
    # Get the relevant data from standings
    # Got rid of gameBack
    XFrame = standings[['gameWon', 'gameLost', 'ptsFor', 'homeWin', 'homeLoss',
                             'awayWin', 'awayLoss', 'ptsScore']]
    # Get the rank from standings
    yFrame = standings[['rank']]
    # Convert all data to float
    XFrame = XFrame.astype(float)

    # Convert to numpy arrays
    X = XFrame.to_numpy()
    scaler.fit(X)
    X = scaler.transform(X)
    # y = yFrame.to_numpy()
    y_svc = {}
    for i in range(0, len(y)):
        y_svc[i] = y[i]
    # Not sure if y should be fit, as each rank is discrete (1, 2, 3, etc)
    # scaler.fit(y)
    # Create KFold
    kf = KFold(n_splits=10)

    model1 = SVC(kernel='linear')
    model1_average_accuracy = 0
    acc1 = {}
    i = 0
    for train, test in kf.split(X):
        # fit data
        model1.fit(X[train], y[train])
        # score accuracy
        acc1[i] = model1.score(X[test], y[test])
        # add accuracy to average
        model1_average_accuracy += acc1[i]
        print('%.2f' % acc1[i])
        i += 1
    # compute average accuracy
    model1_average_accuracy /= 10
    print("Average accuracy " + '%.2f' % str(model1_average_accuracy))

main()
