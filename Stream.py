import pickle
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
st.set_option('deprecation.showPyplotGlobalUse', False)

model=pickle.load(open("xgb.pkl", "rb"))

def main():
    html_temp = """
                <div style="padding:1.5px">
                    <h1 style="color:black;text-align:center;">Big Friday Sales Prediction App</h1>
                </div><br>"""
    st.markdown(html_temp,unsafe_allow_html=True)  
    user_id=st.text_input("Enter User_Id ")
    product_id=st.text_input("Enter Product_Id ")
    gender=st.selectbox("Gender", ["F", "M"])
    age=st.selectbox("Age", ["0-17", "55+","26-35","46-50","51-55","36-45","18-25"])
    Occupation=st.slider("Occupation", 1,20,1)
    city=st.selectbox("City", ["A", "B","C"])
    staycurrent=st.selectbox("Stay In Current City In Years", ["2", "4+","3","1","0"])
    maritalstatus=st.selectbox("Marital Status", [0, 1])
    product1=st.slider("Product Category 1", 1, 20, 1)
    product2=st.slider("Product Category 2", 1, 20, 1)
    product3=st.slider("Product Category 3", 1, 20, 1)
    df = pd.DataFrame(data=[[gender, age, Occupation, city, staycurrent, maritalstatus, product1, product2, product3]], 
                      columns=['Gender', 'Age', 'Occupation', 'City_Category',
       'Stay_In_Current_City_Years', 'Marital_Status', 'Product_Category_1',
       'Product_Category_2', 'Product_Category_3'])

    if st.button("Predict"):
        prediction = model.predict(df)
        st.success(prediction)

if __name__ == "__main__":
    main()