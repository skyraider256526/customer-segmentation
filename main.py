# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(809, 597)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setEnabled(True)
        self.tabWidget.setGeometry(QtCore.QRect(149, 11, 621, 511))
        self.tabWidget.setMinimumSize(QtCore.QSize(521, 0))
        self.tabWidget.setToolTipDuration(1)
        self.tabWidget.setObjectName("tabWidget")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.button_summary = QtWidgets.QPushButton(self.tab_3)
        self.button_summary.setGeometry(QtCore.QRect(470, 60, 107, 28))
        self.button_summary.setObjectName("button_summary")
        self.lineEdit_summary = QtWidgets.QLineEdit(self.tab_3)
        self.lineEdit_summary.setGeometry(QtCore.QRect(32, 60, 421, 28))
        self.lineEdit_summary.setObjectName("lineEdit_summary")
        self.tableWidget = QtWidgets.QTableWidget(self.tab_3)
        self.tableWidget.setGeometry(QtCore.QRect(30, 140, 551, 291))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)
        self.label_3 = QtWidgets.QLabel(self.tab_3)
        self.label_3.setGeometry(QtCore.QRect(40, 100, 89, 20))
        self.label_3.setObjectName("label_3")
        self.label_6 = QtWidgets.QLabel(self.tab_3)
        self.label_6.setGeometry(QtCore.QRect(150, 10, 341, 31))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.tabWidget.addTab(self.tab_3, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.tab_2)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(30, 80, 561, 221))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.lineEdit_model = QtWidgets.QLineEdit(self.verticalLayoutWidget_2)
        self.lineEdit_model.setObjectName("lineEdit_model")
        self.horizontalLayout_3.addWidget(self.lineEdit_model)
        self.button_model = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.button_model.setObjectName("button_model")
        self.horizontalLayout_3.addWidget(self.button_model)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.lineEdit_data_test = QtWidgets.QLineEdit(self.verticalLayoutWidget_2)
        self.lineEdit_data_test.setObjectName("lineEdit_data_test")
        self.horizontalLayout_4.addWidget(self.lineEdit_data_test)
        self.button_data_test = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.button_data_test.setObjectName("button_data_test")
        self.horizontalLayout_4.addWidget(self.button_data_test)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.label_2 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_2.addWidget(self.label_2)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.listWidget_data_test = QtWidgets.QListWidget(self.verticalLayoutWidget_2)
        self.listWidget_data_test.setEnabled(False)
        self.listWidget_data_test.setMaximumSize(QtCore.QSize(300, 300))
        self.listWidget_data_test.setBaseSize(QtCore.QSize(300, 200))
        self.listWidget_data_test.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.listWidget_data_test.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.listWidget_data_test.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectItems)
        self.listWidget_data_test.setObjectName("listWidget_data_test")
        self.horizontalLayout_5.addWidget(self.listWidget_data_test)
        self.button_drop_2 = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.button_drop_2.setEnabled(False)
        self.button_drop_2.setObjectName("button_drop_2")
        self.horizontalLayout_5.addWidget(self.button_drop_2)
        self.verticalLayout_2.addLayout(self.horizontalLayout_5)
        self.button_run = QtWidgets.QPushButton(self.tab_2)
        self.button_run.setGeometry(QtCore.QRect(480, 400, 107, 28))
        self.button_run.setObjectName("button_run")
        self.label_5 = QtWidgets.QLabel(self.tab_2)
        self.label_5.setGeometry(QtCore.QRect(160, 30, 341, 31))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.tabWidget.addTab(self.tab_2, "")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.tab)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(30, 80, 561, 200))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.lineEdit_data_train = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.lineEdit_data_train.setEnabled(True)
        self.lineEdit_data_train.setToolTip("")
        self.lineEdit_data_train.setText("")
        self.lineEdit_data_train.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.lineEdit_data_train.setObjectName("lineEdit_data_train")
        self.horizontalLayout.addWidget(self.lineEdit_data_train)
        self.button_data_train = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.button_data_train.setEnabled(True)
        self.button_data_train.setObjectName("button_data_train")
        self.horizontalLayout.addWidget(self.button_data_train)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.listWidget_data_train = QtWidgets.QListWidget(self.verticalLayoutWidget)
        self.listWidget_data_train.setEnabled(False)
        self.listWidget_data_train.setMaximumSize(QtCore.QSize(300, 300))
        self.listWidget_data_train.setBaseSize(QtCore.QSize(300, 200))
        self.listWidget_data_train.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.listWidget_data_train.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.listWidget_data_train.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectItems)
        self.listWidget_data_train.setObjectName("listWidget_data_train")
        self.horizontalLayout_2.addWidget(self.listWidget_data_train)
        self.button_drop = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.button_drop.setEnabled(False)
        self.button_drop.setObjectName("button_drop")
        self.horizontalLayout_2.addWidget(self.button_drop)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.button_train = QtWidgets.QPushButton(self.tab)
        self.button_train.setGeometry(QtCore.QRect(480, 410, 107, 28))
        self.button_train.setObjectName("button_train")
        self.label_4 = QtWidgets.QLabel(self.tab)
        self.label_4.setGeometry(QtCore.QRect(160, 30, 341, 31))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.tabWidget.addTab(self.tab, "")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(30, 210, 101, 71))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(30, 320, 101, 81))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(30, 110, 101, 71))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton.sizePolicy().hasHeightForWidth())
        self.pushButton.setSizePolicy(sizePolicy)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/home.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton.setIcon(icon)
        self.pushButton.setIconSize(QtCore.QSize(50, 50))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2.raise_()
        self.pushButton_3.raise_()
        self.pushButton.raise_()
        self.tabWidget.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 809, 25))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(2)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Customer Segmentation"))
        self.tabWidget.setToolTip(_translate("MainWindow", "Dataset location"))
        self.button_summary.setText(_translate("MainWindow", "Browse"))
        self.lineEdit_summary.setPlaceholderText(_translate("MainWindow", "Enter dataset location"))
        self.label_3.setText(_translate("MainWindow", "Summary:"))
        self.label_6.setText(_translate("MainWindow", "Generate Summary"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "Page"))
        self.lineEdit_model.setPlaceholderText(_translate("MainWindow", "Enter model loaction"))
        self.button_model.setText(_translate("MainWindow", "Browse"))
        self.lineEdit_data_test.setPlaceholderText(_translate("MainWindow", "Enter dataset location"))
        self.button_data_test.setText(_translate("MainWindow", "Browse"))
        self.label_2.setText(_translate("MainWindow", "Select colums like ID, date, etc to remove"))
        self.button_drop_2.setText(_translate("MainWindow", "Drop"))
        self.button_run.setText(_translate("MainWindow", "Run"))
        self.label_5.setText(_translate("MainWindow", "Run a model"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Tab 2"))
        self.lineEdit_data_train.setPlaceholderText(_translate("MainWindow", "Dataset Location"))
        self.button_data_train.setText(_translate("MainWindow", "Browse"))
        self.label.setText(_translate("MainWindow", "Select colums like ID, date, etc to remove"))
        self.button_drop.setText(_translate("MainWindow", "Drop"))
        self.button_train.setText(_translate("MainWindow", "Train"))
        self.label_4.setText(_translate("MainWindow", "Create a model"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Tab 1"))
        self.pushButton_2.setText(_translate("MainWindow", "Run"))
        self.pushButton_3.setText(_translate("MainWindow", "Summary"))
        self.pushButton.setText(_translate("MainWindow", "Create"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
