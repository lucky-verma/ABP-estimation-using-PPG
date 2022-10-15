import streamlit as st
import os
import numpy as np
import pandas as pd
from helper_functions import *
from models import UNetDS64, MultiResUNet1D
from predict_test import predict_test_data
from evaluate import predicting_ABP_waveform, regression_plot, bland_altman_plot, evaluate_BHS_Standard, evaluate_AAMI_Standard

# Set the page title
st.set_page_config(page_title='PPG2ABP',
                   page_icon=':heartpulse:',
                   layout='wide',
                   initial_sidebar_state='auto')

hide_streamlit_style = """
            <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# # Set the page title
# st.title('PPG2ABP')

# Set the page subtitle
st.markdown('## Analysis of PPG signals')

# Set the page description
st.markdown('''Translating Photoplethysmogram (PPG) Signals to Arterial Blood Pressure (ABP) Waveforms using Fully Convolutional Neural Networks''')

# If test_output.p is not present, predict the outputs for test data
if not os.path.exists('test_output.p'):
    # Button to generate predictions on tes data
    if st.button('Generate Predictions'):
        with st.spinner('Fetching and processing values.....'):
            predict_test_data()
            st.success('Predictions generated successfully!')

# Get number of test samples
test_output = open('test_output.p', 'rb')
num_test_samples = len(pickle.load(test_output))

# If test_output.p is present, load the predictions
if os.path.exists('test_output.p'):
    # Set the option to select the plotting and evaluation method
    option = st.radio('Select the plotting or evaluation ', ('Plotting', 'Evaluation'), horizontal=True)

    if option == 'Plotting':
        # Set the slider to select the sample to be evaluated
        sample = st.slider('Select the sample to be evaluated', min_value=0, max_value=num_test_samples, value=0, step=1)
        with st.spinner('Fetching and processing values.....'):
            st.pyplot(predicting_ABP_waveform(indix=sample))

    if option == 'Evaluation':
        # Set the option to select the evaluation method
        AbsoluteError, BlandAtman, Regression = st.tabs(['Absolute Error', 'Bland-Altman', 'Regression'])

        with AbsoluteError:
            # spinner to show the progress
            with st.spinner('Fetching and processing values.....'):
                fig_ae, dbp_ae, map_ae, sbp_ae = evaluate_BHS_Standard()
                st.pyplot(fig_ae)

                # BHS Standard Evaluation
                st.markdown('''#####  BHS-Metric   ''')
                df_ae = pd.DataFrame({' <= 5mmHg': [str(round(dbp_ae[0], 1)) + '%', str(round(map_ae[0], 1)) + '%', str(round(sbp_ae[0], 1)) + '%'],
                                        ' <= 10mmHg': [str(round(dbp_ae[1], 1)) + '%', str(round(map_ae[1], 1)) + '%', str(round(sbp_ae[1], 1)) + '%'],
                                        ' <= 15mmHg': [str(round(dbp_ae[2], 1)) + '%', str(round(map_ae[2], 1)) + '%', str(round(sbp_ae[2], 1)) + '%'],})

                df_ae.index = ['DBP', 'MAP', 'SBP']
                st.table(df_ae)

        with BlandAtman:
            # spinner to show the progress
            with st.spinner('Fetching and processing values.....'):
                fig_ba, dbp_ba, map_ba, sbp_ba = bland_altman_plot()
                st.pyplot(fig_ba)
                st.markdown(f'''DBP: {dbp_ba} ''')
                st.markdown(f'''MAP: {map_ba} ''')
                st.markdown(f'''SBP: {sbp_ba} ''')

        with Regression:
            # spinner to show the progress
            with st.spinner('Fetching and processing values.....'):
                fig_rg, dbp_rg, map_rg, sbp_rg = regression_plot()
                st.pyplot(fig_rg)
                st.markdown(f'''DBP: {dbp_rg} ''')
                st.markdown(f'''MAP: {map_rg} ''')
                st.markdown(f'''SBP: {sbp_rg} ''')
