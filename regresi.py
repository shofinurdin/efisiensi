import streamlit as st 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from joblib import load
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from PIL import Image
import io
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor


data_lengkap=pd.read_excel('data_regresi/hasil_efisiensi_klaster.xlsx')
X=data_lengkap.drop(columns=['KD_KPP','EFISIENSI_IO','EFISIENSI_OO'])
y=data_lengkap['EFISIENSI_OO']

def run_regresi_app():
    st.write('Halaman Regresi')
    with st.expander('Gradient Boosting Regresor'):
        col1,col2 = st.columns([1,1])
        with col1:
            st.write('GBR')
            mse_scores=[]
            clf = GradientBoostingRegressor()
            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            for train_index, test_index in kf.split(X):
                    X_train, X_test = X.iloc[train_index].values, X.iloc[test_index].values
                    y_train, y_test = y.iloc[train_index].values, y.iloc[test_index].values
                    clf.fit(X_train, y_train.ravel())
                    y_pred = clf.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    mse_scores.append(mse)
            # st.write(f'mse score :{np.mean(mse_scores)}')
            st.write("MSE Score : {:.4f}".format(np.mean(mse_scores)))
            
        with col2:
            st.write('GA-GBR')
            # image = open('gambar/f11a.png', 'rb')
            # st.image(image, caption='Nama Gambar', use_column_width=True)
            st.write('MSE Score: 0,0052')
            with open('gambar/f11a.png', 'rb') as f:
                image = Image.open(io.BytesIO(f.read()))

                # Menampilkan gambar menggunakan st.image()
                st.image(image, caption='Genetic Algorithm Gradient Boosting Regresor', use_column_width=True)
    
    
    with st.expander('Support Vector Regressor'):
        col1,col2 = st.columns([1,1])
        with col1:
            st.write('SVR')
            kf = KFold(n_splits=5)
            # Perform KFold cross-validation
            mse_svr = []
            for train_index, test_index in kf.split(X):
                X_train, X_test = X.iloc[train_index].values, X.iloc[test_index].values
                y_train, y_test = y.iloc[train_index].values, y.iloc[test_index].values
                model_svr = SVR()
                model_svr.fit(X_train, y_train.ravel())
                y_pred = model_svr.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                mse_svr.append(mse)
               
            st.write("MSE Score : {:.4f}".format(np.mean(mse_svr)))
        with col2:
            st.write('GA-SVR')
            st.write('MSE Score: 0.0037')
            with open('gambar/f12a.png', 'rb') as f:
                image = Image.open(io.BytesIO(f.read()))

                # Menampilkan gambar menggunakan st.image()
                st.image(image, caption='Genetic Algorithm Gradient Boosting Regresor', use_column_width=True)

    with st.expander('Random Forest Regresor'):
        col1,col2 = st.columns([1,1])
        with col1:
            st.write('RFR')
            model_rf_1 = RandomForestRegressor()
            kf = KFold(n_splits=5, shuffle=True)
            mse_rfr = []
            
            for train_index, test_index in kf.split(X):
                X_train, X_test = X.iloc[train_index].values, X.iloc[test_index].values
                y_train, y_test = y.iloc[train_index].values, y.iloc[test_index].values

                model_rf_1.fit(X_train, y_train.ravel())
                y_pred = model_rf_1.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                mse_rfr.append(mse)
            st.write("MSE Score : {:.4f}".format(np.mean(mse_rfr)))    
        with col2:
            st.write('GA-RFR')
            st.write('MSE Score: 0.0051')
            with open('gambar/f12b.png', 'rb') as f:
                image = Image.open(io.BytesIO(f.read()))

                # Menampilkan gambar menggunakan st.image()
                st.image(image, caption='Genetic Algorithm Gradient Boosting Regresor', use_column_width=True)


    with st.expander('Multilayer Perceptron'):
        col1,col2 = st.columns([1,1])
        with col1:
            st.write('MLPR')
            st.write('MSE Score: 0.0144')
        with col2:
            st.write('GA-MLPR')
            st.write('MSE Score: 0.0035')
            with open('gambar/f11b.png', 'rb') as f:
                image = Image.open(io.BytesIO(f.read()))

                # Menampilkan gambar menggunakan st.image()
                st.image(image, caption='Genetic Algorithm Multilayer Perceptron Regresor', use_column_width=True)
