#IMPORTS & PACKAGES
import pandas as pd
import numpy as np
import os
from sklearn import linear_model
import math
from sqlalchemy import create_engine
import copy

class dataFrame():
    def __init__(self, dataSize):
        #INITIALIZING DATAFRAME
        self.engine = create_engine("mysql+pymysql://LOLread:"+'LaberLabsLOLquery'+"@lolsql.stat.ncsu.edu/lol2019")
        self.df = pd.read_sql(f'SELECT * from match_player limit {dataSize}',self.engine)
        self.dataSize = dataSize
        self.chid_df = pd.read_csv('id_table.csv')
        self.chid_map = dict(zip(self.chid_df['championId'].tolist(), self.chid_df['name'].tolist()))

    #Select Columns of different features
    def column_selection(self, column_list):
        for column in column_list:
            try:
                assert(type(column) == str)
            except Exception:
                print("Columns Must be Strings")

        #Creating Instance of columns to be used
        self.useful_columns = column_list
        self.colFlag = True

    #Select Champion Rows Based on ID
    def row_selection(self, *argv):
        self.champion_selected = argv[0]
        self.champion_name = self.chid_map[argv[0]]
        self.useful_rows = self.df['championId'] == argv[0]
        self.rowFlag = True

    #Resize DF after initializing cols and rows to be used
    def resize_df(self, normalize = False):
        #row and col must be selected before using
        if self.rowFlag and self.colFlag:
            self.df = self.df.loc[self.useful_rows, self.useful_columns]
        else:
            print("ROWS AND COLS ARE NOT SELECTED")

    def normalize_cols(self, cols_norm, cols_rest):
        #assume using min-max normalization
        #(self.df-self.df.mean())/self.df.std() is for standard normalization
        df_norm = self.df.loc[self.useful_rows, cols_norm]
        df_rest = self.df.loc[self.useful_rows, cols_rest]
        df_norm = (df_norm-df_norm.min())/(df_norm.max()-df_norm.min())
        self.df = pd.concat([df_norm, df_rest], axis=1, sort=False)

    def printDataHead(self):
        print(self.df.head)

class Regressor(object):
    def __init__(self, regressorType):
        self.regressorType = regressorType
        if regressorType == "ridge":
            self.regModel = linear_model.Ridge(alpha=0.5)
        elif regressorType == "logistic":
            self.regModel = linear_model.LogisticRegression(fit_intercept=True,penalty='l2')
        elif regressorType == "linear":
            self.regModel = linear_model.LinearRegression()
        elif regressorType == "lasso":
            self.regModel = linear_model.MultiTaskLasso(alpha=0.1)
        else:
            print("Regressor Not Initialized")

    def obtainX(self, model, x_list):
        self.column_x = x_list
        self.x_df = model.df.loc[:,x_list]

    def obtainY(self, model, y_list):
        self.colun_y = y_list
        self.y_df = model.df.loc[:,y_list]

    def initialize_fit(self, model, x_list, y_list):
        self.champion_selected = model.champion_selected
        self.champion_name = model.champion_name
        self.data_size = model.dataSize
        self.obtainX(model, x_list)
        self.obtainY(model, y_list)

    def fit_data(self):
        self.regModel.fit(self.x_df, self.y_df)
        return self.regModel.coef_

    def print_results(self):
        print(f'The Regressor Selected: {self.regressorType}')
        print(f'The Champion Selected: {self.champion_selected, self.champion_name}')
        print(f'The Result Based on Samples: {self.data_size}')
        print('Parameters: ')
        for i in range(len(self.column_x)):
            print(f'{self.column_x[i]}: {self.regModel.coef_[0][i]:.3f}')
        print('-------------***-------------')



def runRegression(model_data, column_x, column_y):
    for reg_type in ['ridge', 'linear', 'lasso', 'logistic']:
        regression = Regressor(reg_type)
        regression.initialize_fit(model_data, column_x, column_y)
        regression.fit_data()
        regression.print_results()


def main():
    column_list = ['totalMinionsKilled', 'totalDamageDealtToChampions','visionScore',  'win']
    column_x = ['totalMinionsKilled', 'totalDamageDealtToChampions','visionScore']
    column_y = ['win']
    model_data = dataFrame(10000)
    print("-->Raw Data Imported")
    model_data.column_selection(column_list)
    model_data.row_selection(13)
    print("-->Row Col Selection Completed")
    model_data.resize_df()
    print("-->Dataframe Resize Completed")
    model_data.normalize_cols(column_x, column_y)
    print("-->Normalization Completed")
    print('-------------***-------------')
    runRegression(model_data, column_x, column_y)


if __name__ == '__main__':
    main()
