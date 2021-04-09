import os
import sys
from os import path

import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd
import seaborn as sns
from joblib import dump, load
from pandas.core.frame import DataFrame
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUiType
from PyQt5.uic.properties import QtWidgets
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from yellowbrick.cluster.elbow import KElbowVisualizer, kelbow_visualizer

ui, _ = loadUiType('main.ui')


class MainApp(QMainWindow, ui):
    def __init__(self):
        super().__init__()
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.HandleButtons()
        self.InitUI()
        #Store dataset to this
        self.data_train = DataFrame()
        self.data_test = DataFrame()
        self.columnsRemove = []
        self.data_cleaned = DataFrame()
        self.train = True

    def InitUI(self):
        self.tabWidget.tabBar().setVisible(False)

        #Disabling remove columns before loading dataset for training
        self.listWidget_data_train.setEnabled(False)

    def HandleButtons(self):
        self.button_data_train.clicked.connect(self.HandleTrainBrowse)
        self.button_data_test.clicked.connect(self.HandleRunBrowse)
        self.button_drop.clicked.connect(self.RemoveColumn)
        self.button_drop_2.clicked.connect(self.RemoveColumn)
        self.button_train.clicked.connect(self.TrainModel)
        self.button_run.clicked.connect(self.RunModel)
        self.pushButton.clicked.connect(self.Open_Create)
        self.pushButton_2.clicked.connect(self.Open_Run)
        self.pushButton_3.clicked.connect(self.Open_Summary)
        self.button_model.clicked.connect(self.HandleModelBrowse)
        self.button_summary.clicked.connect(self.Summary)

    def GetLocation(self, operation: str, filter: str, caption: str) -> str:
        ''' Get file location either save or open file '''
        if operation == 'open':
            return QFileDialog.getOpenFileName(self,
                                               caption=caption,
                                               directory='.',
                                               filter=filter)[0].strip()
        elif operation == 'save':
            return QFileDialog.getSaveFileName(self,
                                               caption=caption,
                                               directory='.',
                                               filter=filter)[0].strip()

    def HandleTrainBrowse(self):
        ## enable browseing to our os , pick save location
        save_location: str = self.GetLocation(operation='open',
                                              caption="Open",
                                              filter="CSV Files(*.csv)")
        print(save_location)
        if (save_location != ''):
            self.lineEdit_data_train.setText(str(save_location))

            #display columns in listWidget
            self.data_train = pd.read_csv(self.lineEdit_data_train.text())
            cols = self.data_train.columns.values.tolist()
            print(cols)
            self.listWidget_data_train.addItems(cols)
            self.listWidget_data_train.setEnabled(True)
            self.button_drop.setEnabled(True)
            self.train = True

    def HandleModelBrowse(self):
        self.model_location = self.GetLocation(operation='open',
                                               caption="Open",
                                               filter="JobLib Files(*.joblib)")
        if (self.model_location != ''):
            self.lineEdit_model.setText(str(self.model_location))

    def HandleRunBrowse(self):
        ## enable browseing to our os , pick save location
        data_location = self.GetLocation(operation='open',
                                         caption="Open",
                                         filter="CSV Files(*.csv)")
        if data_location != '':
            self.lineEdit_data_test.setText(str(data_location))
            #display columns in listWidget
            self.data_test = pd.read_csv(self.lineEdit_data_test.text())
            cols = self.data_test.columns.values.tolist()
            print(cols)
            self.listWidget_data_test.addItems(cols)
            self.listWidget_data_test.setEnabled(True)
            self.button_drop_2.setEnabled(True)
            self.train = False

    def RemoveColumn(self):
        if (self.train):
            items = self.listWidget_data_train.selectedItems()
            list = self.listWidget_data_train
            data = self.data_train
        else:
            items = self.listWidget_data_test.selectedItems()
            list = self.listWidget_data_test
            data = self.data_test
        if items is None:
            return
        reply = QMessageBox.question(
            self, "Drop",
            "Remove`{0}'?".format(' '.join(map(lambda item: item.text(),
                                               items))),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            for item in items:
                row = list.row(item)
                item = list.takeItem(row)
                self.columnsRemove.append(item.text())
                del item
            #Delete from dataframe only in training
            self.data_cleaned = data.drop(columns=self.columnsRemove,
                                          inplace=self.train)

    def TrainModel(self):
        print(self.data_train.columns)
        self.listWidget_data_train.clear()
        self.columnsRemove.clear()
        save_location = self.GetLocation(operation='save',
                                         caption="Save as",
                                         filter="JobLib Files(*.joblib)")
        if save_location != '':
            print(save_location, 'model train start')
            #train model
            self.data_train.dropna(inplace=True)
            self.data_train.drop_duplicates(inplace=True)
            X = pd.get_dummies(self.data_train)
            kmeans = KMeans(init='k-means++',
                            max_iter=300,
                            n_init=10,
                            random_state=4)
            scaler = MinMaxScaler()
            scaled_features = scaler.fit_transform(X)
            visualizer = KElbowVisualizer(kmeans,
                                          k=(4, 12),
                                          metric='silhouette',
                                          timings=False)

            visualizer.fit(X)

            if (not visualizer.elbow_value_):
                clusterValue = 3
            else:
                clusterValue = visualizer.elbow_value_
            kmeans = KMeans(max_iter=300,
                            n_init=10,
                            random_state=4,
                            n_clusters=clusterValue)
            print(clusterValue)
            kmeans.fit(scaled_features)
            #save model
            dump(kmeans, save_location + '.joblib')
            print('model train done')

    def RunModel(self):
        print(self.data_cleaned.columns)
        self.listWidget_data_test.clear()
        self.model = load(self.model_location)
        self.columnsRemove.clear()
        self.data_cleaned.dropna(inplace=True)
        self.data_cleaned.drop_duplicates(inplace=True)
        X = pd.get_dummies(self.data_cleaned)
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(X)
        y_means = self.model.predict(scaled_features)
        self.data_cleaned['Cluster'] = y_means
        self.data_cleaned.to_csv('output.csv')

    def Summary(self):
        data_location = self.GetLocation('open', 'CSV Files(*.csv)', 'Open')
        if data_location != '':
            self.lineEdit_summary.setText(data_location)
            df = pd.read_csv(data_location)
            summary_df = df.describe()

            #Row count
            row = summary_df.shape[0]
            self.tableWidget.setRowCount(row)

            #Column count
            column = summary_df.shape[1]
            self.tableWidget.setColumnCount(column)

            self.tableWidget.setHorizontalHeaderLabels(
                summary_df.columns.values.tolist())
            self.tableWidget.setVerticalHeaderLabels(
                summary_df.index.values.tolist())
            print(row, column)
            for i in range(row):
                for j in range(column):
                    self.tableWidget.setItem(
                        i, j, QTableWidgetItem(str(summary_df.iloc[i, j])))
            self.tableWidget.resizeColumnsToContents()
            self.tableWidget.resizeRowsToContents()
            self.tableWidget.setEnabled(True)


################################################
###### UI CHanges Methods

    def Open_Create(self):
        self.tabWidget.setCurrentIndex(2)

    def Open_Run(self):
        self.tabWidget.setCurrentIndex(1)

    def Open_Summary(self):
        self.tabWidget.setCurrentIndex(0)


def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec()


if __name__ == '__main__':
    main()
