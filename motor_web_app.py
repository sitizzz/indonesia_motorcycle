import streamlit as st
import numpy as np
import pickle
import base64

#loading the models
model_honda = pickle.load(open('ros_reg_rf_honda.pkl', 'rb'))
model_yamaha = pickle.load(open('yamaha_smote.pkl','rb'))
model_suzuki = pickle.load(open('finalized_suzuki.pkl','rb'))

#creating a function for prediction
def predict_honda(input_data):
    honda = [v[0] for v in input_data]           
    combined_data = input_data[2] + input_data[3]
    
    #changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(combined_data)
    
    #reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
    predict_honda = model_honda.predict(input_data_reshaped)
    return f'Estimated price for **{honda[0]} {honda[1]}** is IDR **{int(predict_honda):,}**'

def predict_yamaha(input_data):
    print(input_data)
    yamaha = [v[0] for v in input_data]           
    combined_data = input_data[2] + input_data[3]
            
    #changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(combined_data)
    
    #reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
    predict_yamaha = model_yamaha.predict(input_data_reshaped)
    return f'Estimated price for **{yamaha[0]} {yamaha[1]}** is IDR **{int(predict_yamaha):,}**'
 
def predict_suzuki(input_data):
    print(input_data)
    suzuki = [v[0] for v in input_data]
    combined_data = input_data[2] + input_data[3]
            
    #changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(combined_data)
    
    #reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
    predict_suzuki = model_suzuki.predict(input_data_reshaped)
    return f'Estimated price for **{suzuki[0]} {suzuki[1]}** is IDR **{int(predict_suzuki):,}**'

def main():
    #creating a title
    st.title('Price Prediction (Indonesia Region)')
    st.subheader('Second Hand Motorcycle')
    
    st.write("Hello, I am Siti Zubaidah, the creator of this web app!")
    st.write("This web application is my observation area to see whether streamlit able to run several machine learning models. It consists of three machine learning models for each brand. Namely: Honda, Suzuki and Yamaha. You can learn the code through my github [here](https://github.com/sitizzz)")
    st.write("If you consider to copy this web app. Kindly give credit to my name and hyperlink this web app.")
    st.write("Learn data science [here](https://www.youtube.com/channel/UCeFBTC1HiIAR3rd11CuCM4A) (Bahasa Indonesia Only)")
    
    #getting the input data from the user
    Merek = st.selectbox('Choose brand: ', ['Honda', 'Suzuki', 'Yamaha'])
             
    if Merek == 'Honda':
        Model_honda = st.selectbox('Choose model: ',['ADV', 'Beat', 'Blade', 'C70', 'CB', 'CBR', 'CRF', 'CS One', 'GL',
           'Genio', 'Mega Pro', 'PCX', 'Rebel','Revo', 'Scoopy', 'Sonic', 'Spacy', 'Supra', 'Tiger', 'Vario', 'Verza'])
        
        Model_dict_honda = {'ADV':0, 'Beat':1, 'Blade':2, 'C70':3, 'CB':4, 'CBR':5, 'CRF':6, 'CS One':7, 'GL':8,
           'Genio':9, 'Mega Pro':10, 'PCX':11, 'Rebel':12,'Revo':13, 'Scoopy':14, 'Sonic':15, 'Spacy':16,
           'Supra':17, 'Tiger':18, 'Vario':19, 'Verza':20}
        
        Modelz_honda = np.zeros(21, dtype=int)
        if Model_honda in Model_dict_honda:
            Modelz_honda[Model_dict_honda[Model_honda]]=1
            Modelz_honda = list(Modelz_honda)
            
        Tahun_honda = st.selectbox('Choose year of assemble: ', list(range(1975,2020)))
        
        #code for prediction
        price=''
        
        #creating a button for prediction
        if st.button('Start Predicting'): 
        
            price = predict_honda([[Merek], [Model_honda], [Tahun_honda], Modelz_honda])
        
    elif Merek == 'Suzuki':
        Model_suzuki = st.selectbox('Choose model: ',['Address', 'GSX / Katana', 'Nex','Satria', 'Shogun', 'Skywave'])
        
        Model_dict_suzuki = {'model_Address':0, 'model_GSX / Katana':1, 'model_Nex':2,
        'model_Satria':3, 'model_Shogun':4, 'model_Skywave':5}
        
        Modelz_suzuki = np.zeros(6, dtype=int)
        if Model_suzuki in Model_dict_suzuki:
            Modelz_suzuki[Model_dict_suzuki[Model_suzuki]]=1
        
        Tahun_suzuki = st.selectbox('Choose year of assemble: ', list(range(1990,2024)))
        
        #code for prediction
        price=''
        
        #creating a button for prediction
        if st.button('Start Predicting'): 
       
            price = predict_suzuki([[Merek], [Model_suzuki], [Tahun_suzuki], list(Modelz_suzuki)])
                
    else:
        Model_yamaha = st.selectbox('Choose model: ',['Aerox', 'Byson', 'F1','F1ZR', 'Fino', 'FreeGo', 'Jupiter',
           'Lexi', 'MT-25', 'MX', 'Mio', 'NMax','Nouvo', 'R15', 'RX', 'RX135', 'Scorpio',
           'Soul', 'V-Xion', 'Vega', 'X-MAX','X-Ride', 'Xabre', 'Xeon', 'YZF'])
        
        Model_dict_yamaha = {'Aerox':0, 'Byson':1, 'F1':2,'F1ZR':3, 'Fino':4, 'FreeGo':5, 'Jupiter':6,
           'Lexi':7, 'MT-25':8, 'MX':9, 'Mio':10, 'NMax':11,'Nouvo':12, 'R15':13, 'RX':14, 'RX135':15, 'Scorpio':16,
           'Soul':17, 'V-Xion':18, 'Vega':19, 'X-MAX':20,'X-Ride':21, 'Xabre':22, 'Xeon':23, 'YZF':24}
        
        Modelz_yamaha = np.zeros(25, dtype=int)
        if Model_yamaha in Model_dict_yamaha:
            Modelz_yamaha[Model_dict_yamaha[Model_yamaha]]=1
            Modelz_yamaha = list(Modelz_yamaha)
        
        Tahun_yamaha = st.selectbox('Choose year of assemble: ', list(range(1992,2020)))
        
        #code for prediction
        price=''
        
        #creating a button for prediction
        if st.button('Start Predicting'):  
        
            price = predict_yamaha([[Merek], [Model_yamaha], [Tahun_yamaha], Modelz_yamaha]) 
         
    st.success(price)
           
    col1, col2, col3 = st.columns(3)
    
    with col1:        
        def get_base64_of_bin_file(bin_file):
            with open(bin_file, 'rb') as f:
                data = f.read()
            return base64.b64encode(data).decode()
        
        def get_img_with_href(local_img_path, target_url):
            bin_str = get_base64_of_bin_file(local_img_path)
            html_code = f'''
                <a href="{target_url}">
                    <img src="data:image/jpg;base64,{bin_str}" width="160" height="125"/>
                </a>'''
            return html_code
        
        st.markdown("**Sponsored by:**")
        jpg_html = get_img_with_href('brs_toped.jpg', 'https://www.tokopedia.com/bungarampaistore')
        
        st.markdown(jpg_html, unsafe_allow_html=True)
    
    with col3:
        st.markdown("**Empowered by:**")       
        jpg_html = get_img_with_href('symbol_logo_basic_color.jpg', 'https://en.apu.ac.jp/home/')
        
        st.markdown(jpg_html, unsafe_allow_html=True)
        
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
