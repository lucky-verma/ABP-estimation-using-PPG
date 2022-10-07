import streamlit as st
import os
import numpy as np
import pandas as pd
from helper_functions import *
from models import UNetDS64, MultiResUNet1D
from predict_test import predict_test_data
from evaluate import predicting_ABP_waveform

# Set the page title
st.set_page_config(page_title='PPG2ABP', page_icon=':heartpulse:', layout='wide', initial_sidebar_state='auto')

# # Set the page title
# st.title('PPG2ABP')

# Set the page subtitle
st.markdown('## A web application for the analysis of PPG signals')

# Set the page description
st.markdown('''Translating Photoplethysmogram (PPG) Signals to Arterial Blood Pressure (ABP) Waveforms using Fully Convolutional Neural Networks''')

# If test_output.p is not present, predict the outputs for test data
if not os.path.exists('test_output.p'):
    # Button to generate predictions on tes data
    if st.button('Generate Predictions'):
        with st.spinner('Fetching and processing values.....'):
            predict_test_data()
            st.success('Predictions generated successfully!')

# If test_output.p is present, load the predictions
if os.path.exists('test_output.p'):
    # Set the slider to select the sample to be evaluated
    sample = st.slider('Select the sample to be evaluated', min_value=0, max_value=27000, value=0, step=1)
    with st.spinner('Fetching and processing values.....'):
        st.pyplot(predicting_ABP_waveform(indix=sample))