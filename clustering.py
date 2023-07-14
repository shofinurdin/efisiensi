import streamlit as st
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import plotly.express as px 

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.decomposition import PCA


# from sklearn_extra.cluster import KMedoids
from sklearn.cluster import OPTICS
import skfuzzy as fuzz
from sklearn.cluster import DBSCAN


@st.cache_data
def load_data(data):
	df = pd.read_csv(data)
	return df

def variabel_input(data):
	v_in= data[['WP_BENDAHARA', 'WP_BADAN','WP_OP_KARYAWAN', 'WP_OP_PENGUSAHA', 'JUMLAH_FPP', 'JUMLAH_AR','REALISASI_ANGGARAN']]
	return v_in

def minmax_scaler(data):
	scaler = MinMaxScaler()
	data_mm = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
	return data_mm

def pca_scale(data):
	pca_model = PCA(n_components=2)
	pca_vin = pca_model.fit_transform(data)
	hasil_pca = pd.DataFrame(pca_vin, columns=['Feature A', 'Feature B'])
	return hasil_pca

def bikin_dataframe(list1, list2,list3):
	series1 = pd.Series(list1)
	series2 = pd.Series(list2)
	series3 = pd.Series(list3)
	df = pd.concat([series1, series2, series3], axis=1)
	return df


def run_cl_app():
	data_awal = load_data('data/data_fix.csv')
	v_in=variabel_input(data_awal)
	vin_scale=minmax_scaler(v_in)
	hasil_pca=pca_scale(vin_scale)
	submenu = st.sidebar.selectbox("Submenu",['Model Awal','Hasil'])
	if submenu == 'Model Awal':
		
		st.write('Model Klastering Awal')

		silhouette_list=[]
		dbi_list=[]

		with st.expander('Tabel'):
			st.dataframe(vin_scale)


		
	    
		with st.expander('Fuzzy C-Means'):
			#k = 5
			#m = 2.0
			x=vin_scale.values
			cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(x.T, 5, 2.0, error=0.005, maxiter=1000)
			labels_fuzz = np.argmax(u, axis=0)
			siluete_fcm = silhouette_score(x, labels_fuzz)
			
			
			st.info(f'Nilai Silhouette : {round(siluete_fcm,2):.2f}')

			silhouette_list.append(siluete_fcm)
			hasil_pca['FCM']=labels_fuzz

			dbi_fcm = davies_bouldin_score(x, labels_fuzz)
			dbi_list.append(dbi_fcm)

			fig, ax = plt.subplots()
			scatter = ax.scatter(hasil_pca['Feature A'], hasil_pca['Feature B'], c=hasil_pca['FCM'])
			legend = ax.legend(*scatter.legend_elements(), title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
			ax.add_artist(legend)
			st.pyplot(fig)

		

	if submenu == 'Hasil':
		st.write('Hasil')

		with st.expander('Fuzzy C-Means'):
			st.write('Fuzzy C-Means')
			x=vin_scale.values
			fcm_model = fuzz.cluster.cmeans(x.T, 3, 2, error=0.005, maxiter=1000)
			cluster_membership = fcm_model[1]
			labels_fuzz_opt = cluster_membership.argmax(axis=0)
			silhouette_avg = silhouette_score(x, labels_fuzz_opt)
			st.info(f'Nilai Silhouette : {round(silhouette_avg,2):.2f}')
			hasil_pca['fcm_opt']=labels_fuzz_opt
			fig, ax = plt.subplots()
			plt.title("FCM Clustering")
			plt.xlabel("PC1")
			plt.ylabel("PC2")
			scatter = ax.scatter(hasil_pca['Feature A'], hasil_pca['Feature B'], c=hasil_pca['fcm_opt'])
			legend = ax.legend(*scatter.legend_elements(), title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
			ax.add_artist(legend)
			st.pyplot(fig)