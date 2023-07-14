import streamlit as st 
import streamlit.components.v1 as stc 
from simulasi_app import run_simulasi_app
from clustering import run_cl_app

st.set_page_config(page_title="ML & DEA",
		   page_icon="⚙️",
		   layout="wide")


html_temp = """
		<div style="background-color:#3872fb;padding:5px;border-radius:10px">
		<h3 style="color:white;text-align:center;font-family:arial;">Penggunaan Machine Learning Pada Data Envelopment Analysis </h3>
		<h3 style="color:white;text-align:center;font-family:arial;">Untuk Pengukuran Efisiensi Kantor Pelayanan Pajak</h3>
		<h3 style="color:white;text-align:center;"></h3>
		<h4 style="color:white;text-align:center;font-family:arial;">--Prototype--</h4>
		</div>
		"""

def main():
	#st.title("ML Web App with Streamlit")
	stc.html(html_temp)

	menu = ["Home","Clustering","Simulasi Prediksi"]
	choice = st.sidebar.selectbox("Menu",menu)

	if choice == "Home":
		st.subheader("Home")
		st.write("""
			#### Penggunaan Machine Learning Pada Data Envelopment Analysis Untuk Pengukuran Efisiensi Kantor Pelayanan Pajak
			
			#### App Content
				- EDA Section: Exploratory Data Analysis of Data
				- Clustering Section: ML Clustering App
				- DEA Section: Data Envelopment Analysis App
				- Regression Section: ML Predictor App

			""")
	elif choice == "Simulasi Prediksi":
		run_simulasi_app()
	else:
		st.subheader("Clustering")
		run_cl_app()
		

if __name__ == '__main__':
	main()