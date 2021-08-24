import streamlit as st
import streamlit.components as stc
import pandas as pd

import base64
import time
timestr = time.strftime("%Y%m%d-%H%M%S")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score


#Sets the layout to full width
st.set_page_config(layout= "wide")

# title of the app
st.title("""
# Sales Prediction based on Marketing Investment

The *RandomForestRegressor* is utilised in this implementation to anticipate sales based on marketing investment. 

Tune the hyperparameters for more accuracy!

""")

#---------------------------------------------------------#

#Model Building
def build_model(df):
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    #Data Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split_size)

    st.markdown('**1.1 Data Splits**')
    st.write("Training Set")
    st.info(X_train.shape)
    st.write('Test Set')
    st.info(X_test.shape)

    st.markdown('**1.2 Variable Details**')
    st.write('Independent variable (X)')
    st.info(list(X.columns))
    st.write('Dependent variable (y)')
    st.info(y.name)

    rf = RandomForestRegressor(n_estimators=parameter_n_estimators, random_state=parameter_random_state,
                               max_features=parameter_max_features, criterion=parameter_criterion,
                               min_samples_split=parameter_min_samples_split,
                               min_samples_leaf=parameter_min_samples_leaf,
                               bootstrap=parameter_bootstrap, oob_score=parameter_oob_score, n_jobs=parameter_n_jobs)
    rf.fit(X_train,y_train)

    st.subheader('2. Model Performance')

    st.markdown("**2.1 Training Set")
    y_pred_train = rf.predict(X_train)
    st.write('Coefficient of determination ($R^2$): ')
    st.info(r2_score(y_train,y_pred_train))

    st.write('Error (MSE or MAE): ')
    st.info(mean_squared_error(y_train,y_pred_train))


    st.markdown("**2.1 Test Set")
    y_pred_test = rf.predict(X_test)
    st.write('Coefficient of determination ($R^2$): ')
    st.info(r2_score(y_test, y_pred_test))

    st.write('Error (MSE or MAE): ')
    st.info(mean_squared_error(y_test, y_pred_test))




    st.subheader('3. Model Parameters')
    st.write(rf.get_params())

#---------------------------------------------------------#

#adding a sidebar
st.sidebar.header("1. Upload File")

#setup file upload
uploaded_file = st.sidebar.file_uploader(label="Upload your CSV file",
                         type=['csv'])

#Sidebar - Specify Parameter settings
with st.sidebar.header("2. Set Parameters"):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)',0.1, 0.9, 0.25, 0.05)

with st.sidebar.subheader("2.1 Learning Parameters"):
    parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)
    parameter_max_features = st.sidebar.selectbox('Max features (max_features)', ('auto', 'sqrt', 'log2'))
    parameter_min_samples_split = st.sidebar.slider(
        'Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
    parameter_min_samples_leaf = st.sidebar.slider(
        'Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

with st.sidebar.subheader('2.2. General Parameters'):
    parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
    parameter_criterion = st.sidebar.selectbox('Performance measure (criterion)', ('mse', 'mae'))
    parameter_bootstrap = st.sidebar.selectbox('Bootstrap samples when building trees (bootstrap)', (True, False))
    parameter_oob_score = st.sidebar.selectbox(
        'Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', (False, True))
    parameter_n_jobs = st.sidebar.selectbox('Number of jobs to run in parallel (n_jobs)', (1, -1))

#---------------------------------------------------------#




#---------------------------------------------------------#
#Main Panel
st.subheader("1. Dataset")

global df
if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown('**1.1 Glimpse of Dataset**')
        st.write(df)
        build_model(df)
else:
    st.info("Awaiting for CSV file to be uploaded.")


#-------------------------------------------------------#








