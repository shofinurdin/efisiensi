import streamlit as st 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from joblib import load


@st.cache_data
def load_data(data):
	df = pd.read_csv(data)
	return df


def pilih_kolom(data):
	data_kolom=data[['WP_BENDAHARA', 'WP_BADAN','WP_OP_KARYAWAN', 'WP_OP_PENGUSAHA', 'JUMLAH_FPP', 'JUMLAH_AR','REALISASI_ANGGARAN', 
			'KEPATUHAN_SPT', 'CAPAIAN_PENERIMAAN','PERTUMBUHAN_PENERIMAAN', 'SP2DK_TERBIT', 'SP2DK_CAIR','PEMERIKSAAN_SELESAI']]
	return data_kolom



def run_simulasi_app():
	st.write('Simulasi Prediksi')
		
	data_awal = load_data('data/data_fix.csv')

def minmax_scaler(data_awal,data_baru):
	scaler = MinMaxScaler()
	scaler.fit_transform(data_awal)
	scaled_new_data = scaler.transform(data_baru)
	return scaled_new_data

def load_model():
	model_gb=load('model/model_mlp.joblib')
	return model_gb

def run_simulasi_app():
	# data_=st.file_uploader("Upload Data Set", type=['csv'])
	data_ = load_data('data/data_fix.csv')
	if data_ is not None:
		data_awal=pd.read_csv('data/data_fix.csv')
		data_kolom = pilih_kolom(data_awal)
		data_baru=np.array([36,4297,31733,4922,7,30,7211834457,103.06,138.82,20.07,1463,1786,185])
		contoh=data_kolom.iloc[:1,:13]
		

		#with st.expander('Data Baru'):
		st.write('Variabel Input :')
		col1,col2,col3,col4=st.columns([1,1,1,1])

		with col1 :
			x1=st.number_input('WP_BENDAHARA',  value=36, min_value=0)
			x5=st.number_input('JUMLAH_FPP',value=7, min_value=0)
		with col2:
			x2=st.number_input('WP_BADAN',  value=4297, min_value=0)
			x6=st.number_input('JUMLAH_AR',  value=30, min_value=0)
			
		with col3:
			x3=st.number_input('WP_OP_KARYAWAN',  value=31733, min_value=0)
			x7=st.number_input('REALISASI_ANGGARAN',  value=7211834457, min_value=0)
		with col4:
			x4=st.number_input('WP_OP_PENGUSAHA',  value=4922, min_value=0)
			
		st.markdown("<hr>", unsafe_allow_html=True)
		st.write('Variabel Output :')
		col1,col2,col3=st.columns([1,1,1])
		with col1:
			x8=st.number_input('KEPATUHAN_SPT',  value=103.06, min_value=0.0)
			x11=st.number_input('SP2DK_TERBIT',  value=1990, min_value=0)
			
		with col2:
			x9=st.number_input('CAPAIAN_PENERIMAAN',  value=138.82, min_value=0.0)
			x12=st.number_input('SP2DK_CAIR',  value=1786, min_value=0)
			
		with col3:
			x10=st.number_input('PERTUMBUHAN_PENERIMAAN',  value=20.07)
			x13=st.number_input('PEMERIKSAAN_SELESAI',  value=185, min_value=0)
			
		st.markdown("<hr>", unsafe_allow_html=True)
		
		if st.button('Proses'):
			data = {'WP_BENDAHARA': x1,
						'WP_BADAN': x2,
						'WP_OP_KARYAWAN': x3,
						'WP_OP_PENGUSAHA': x4,
						'JUMLAH_FPP': x5,
						'JUMLAH_AR': x6,
						'REALISASI_ANGGARAN': x7,
						'KEPATUHAN_SPT': x8,
						'CAPAIAN_PENERIMAAN': x9,
						'PERTUMBUHAN_PENERIMAAN': x10,
						'SP2DK_TERBIT': x11,
						'SP2DK_CAIR': x12,
						'PEMERIKSAAN_SELESAI': x13}
		
			df_input=pd.DataFrame(data,index=[0])
			st.write('inputan :')
			st.dataframe(df_input)
				
			model=load('model/model_mlp.joblib')
			
			transformer = MinMaxScaler()
			transformer.fit_transform(data_kolom)
			hasil_minmax = transformer.transform(df_input.iloc[:1,:])
			col_names = ['WP_BENDAHARA', 'WP_BADAN','WP_OP_KARYAWAN', 'WP_OP_PENGUSAHA', 'JUMLAH_FPP', 'JUMLAH_AR','REALISASI_ANGGARAN', 
			'KEPATUHAN_SPT', 'CAPAIAN_PENERIMAAN','PERTUMBUHAN_PENERIMAAN', 'SP2DK_TERBIT', 'SP2DK_CAIR','PEMERIKSAAN_SELESAI']
			df_scal = pd.DataFrame(hasil_minmax, columns=col_names)
			# hasil_minmax.columns = col_names
			st.write('hasil scaling :')
			
			st.dataframe(df_scal)
			hasil_predict=model.predict(hasil_minmax)
			st.info(f'Hasil prediksi nilai Efisiensi : {round(hasil_predict[0],2):.2f}')
			if round(hasil_predict[0],2)>1.00:
				st.success('sudah mencapai efisien tetapi Variabel input masih bisa ditambah atau variabel output masih bisa dikurangi')
			elif round(hasil_predict[0],2)<1.00:
				st.success('Belum efisien variabel input masih bisa dikurangi atau variabel outuput masih bisa ditambah')
			else:
				st.success('sudah mencapai efisien')
		
	else:
		st.write('data tidak ada')	
