#importar librerias
import streamlit as st
import pickle
import pandas as pd

#Extrar los archivos pickle


filename = ('log_reg.pkl')
log_reg = pickle.load(open(filename, 'rb'))

#funcion para clasificar las plantas 
def classify(num):
    if num == 0:
        return 'male'
    else:
        return 'female'

def main():
    #titulo
    st.title('Modelo de prueba para el curso DevOps')
    #titulo de sidebar
    st.sidebar.header('Introducir los valores')

    #funcion para poner los parametrso en el sidebar
    def user_input_parameters():
        bill_depth = st.sidebar.slider('bill_depth', 13.1, 21.5, 15.4)
        flipper_lenght_mm = st.sidebar.slider('flipper_lenght_mm', 172, 231, 200)
        body_mass_g = st.sidebar.slider('body_mass_g', 2700, 6300, 5000)
     
        data = {'bill_depth': bill_depth,
                'flipper_lenght_mm': flipper_lenght_mm,
                'body_mass_g': body_mass_g }
        features = pd.DataFrame(data, index=[0])
        return features

    df = user_input_parameters()


    if st.button('RUN'):
       
            st.success(classify(log_reg.predict(df)))
      


if __name__ == '__main__':
    main()
    