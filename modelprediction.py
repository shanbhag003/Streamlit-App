import streamlit as st
import pandas as pd
import pickle

import base64 #to code & encode into excel format
import time #for file-naming convention
timestr = time.strftime("%Y%m%d-%H%M%S")


#Sets the layout to full width
st.set_page_config(layout= "wide")

# title of the app
st.title("""
# Main Prediction

Upload the data set for predictions. 

Upload only those features that were used for training the model.

""")

#un-pickling the pickle file
model = pickle.load(open('myfile.pkl','rb'))


#adding a sidebar
st.sidebar.header("1. Upload CSV File")

#setup file upload
uploaded_file = st.sidebar.file_uploader(label="Upload your file in CSV format",
                         type=['csv'])

#-------------------------------------------------------------------#

#Prediction

def predict_results(df):
    X = df

    st.markdown('**Variable Details**')
    st.write('Dataset contains following features: ')
    st.info(list(X.columns))

    prediction = model.predict(X)
    prediction = pd.DataFrame(prediction, columns =["predicted_sales"])
    final_df = pd.concat([X,prediction],axis=1)

    return final_df

#-----------------------------------------------------------------#

#Download predicted results

def filedownload(df):
    csvfile = df.to_csv(index=False)
    b64 = base64.b64encode(csvfile.encode()).decode() # strings <-> bytes conversions
    new_filename = "new_csv_file_{}_.csv".format(timestr)
    href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Download CSV File with Predicted results.</a>'
    return href


#----------------------------------------------------------------#

#Main Panel

st.subheader("Dataset")

global df
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**Glimpse of Dataset**')
    st.write(df)
    pred = predict_results(df)
    st.markdown(filedownload(pred), unsafe_allow_html=True)
else:
    st.info("Awaiting for CSV file to be uploaded.")
















