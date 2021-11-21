# Reference: https://github.com/ARGULASAISURAJ/Stock-Price-visualisation-Web-App/blob/master/Visualise_Stock_market_Prices_Google_App.py
# Reference: host to heroku https://medium.com/analytics-vidhya/how-to-deploy-a-streamlit-app-with-heroku-5f76a809ec2e
# Reference: https://hayirici.medium.com/stock-price-prediction-using-machine-learning-algorithms-961e6dce74f2

123

import yfinance as yf
import streamlit as st
import datetime
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests
from pandas import DataFrame
import matplotlib.dates as mdates
import nltk
import re
import string
import time
import json
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import TimeSeriesSplit 
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR 
import seaborn as sns

st.set_page_config(layout="wide")

st.write('# Stock Price Prediction')
st.write('### Please enter the stock name and start date you want before using the function. Enjoy!')

option = st.text_input('Stock name')

st.write('Your selection: ', option)
tickerSymbol = option
tickerData = yf.Ticker(tickerSymbol)


start='2010-07-23'
start=st.text_input('Start Date format YYYY-MM-DD')
end=datetime.datetime.today().strftime ('%Y-%m-%d')
#get data on this ticker
st.write('From given Start date', start ,'to current Date', end)
#define the ticker symbol

# -------------


#get the historical prices for this ticker
#reference: https://github.com/mediasittich/Predicting-Stock-Prices-with-Linear-Regression/blob/master/Predicting%20Stock%20Prices%20with%20Linear%20Regression.ipynb
if option:
    tickerDf = yf.download(tickerSymbol, period='1d', start=start, end=end)
    testDf = yf.download(tickerSymbol,start = '2012-01-01').dropna()

@st.cache
def convert_df(df):
# IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')

if st.checkbox('Stock Company Info'):
    st.write('## Company Info')
    st.write(pd.json_normalize(tickerData.info))
    csv3 = convert_df(pd.json_normalize(tickerData.info))
    st.download_button(
        label="Download data as CSV",
        data=csv3,
        file_name='Company_Info.csv',
        mime='text/csv',
    )

    st.write('## Cash Flow Per Year')
    tickerData.cashflow
    csv4 = convert_df(tickerData.cashflow)
    st.download_button(
        label="Download data as CSV",
        data=csv4,
        file_name='Company_Cash_Flow_Per_Year.csv',
        mime='text/csv',
    )
    

if st.checkbox('Show stock data'):
    st.write(tickerDf)  
    csv2 = convert_df(tickerDf)
    st.download_button(
        label="Download data as CSV",
        data=csv2,
        file_name=option + ' data.csv',
        mime='text/csv',
    )


#--------------------
if st.checkbox('Show Chart'):
    option = st.selectbox(
     'Please select type of chart',
     ('Line Chart', 'Bar Chart', 'Area Chart'))

    if option == 'Line Chart':
        st.write("""
        ## Closing Price
        """)
        st.line_chart(tickerDf.Close)

        st.write("""
        ## Opening Price
        """)
        st.line_chart(tickerDf.Open)

        st.write("""
        ## High Price
        """)
        st.line_chart(tickerDf.High)

        st.write("""
        ## Low Price
        """)
        st.line_chart(tickerDf.Low)

        st.write("""
        ## Volume
        """)
        st.line_chart(tickerDf.Volume)

    elif option == 'Area Chart':
        st.write("""
        ## Closing Price
        """)
        st.area_chart(tickerDf.Close)

        st.write("""
        ## Opening Price
        """)
        st.area_chart(tickerDf.Open)

        st.write("""
        ## High Price
        """)
        st.area_chart(tickerDf.High)

        st.write("""
        ## Low Price
        """)
        st.area_chart(tickerDf.Low)

        st.write("""
        ## Volume
        """)
        st.area_chart(tickerDf.Volume)
    
    elif option == 'Bar Chart':
        st.write("""
        ## Closing Price
        """)
        st.bar_chart(tickerDf.Close)

        st.write("""
        ## Opening Price
        """)
        st.bar_chart(tickerDf.Open)

        st.write("""
        ## High Price
        """)
        st.bar_chart(tickerDf.High)

        st.write("""
        ## Low Price
        """)
        st.bar_chart(tickerDf.Low)

        st.write("""
        ## Volume
        """)
        st.bar_chart(tickerDf.Volume)
    
    else:
        print('error')


if st.checkbox('Risk and Return Analysis'):
    st.write("### We use price start from 2012 until yesterday historical data if not value too small")

    st.write('Risk and return')
    #This function is calculated annual risk and return for stocks
    df_adj_close = testDf['Adj Close']
    df_adj_close = df_adj_close.to_frame()
    ret = df_adj_close.pct_change()
    def ann_risk_return(returns_df):
        summary = returns_df.agg(["mean", "std"]).T
        summary.columns = ["Return", "Risk"]
        summary.Return = summary.Return*252
        summary.Risk = summary.Risk * np.sqrt(252)
        return summary
    summary = ann_risk_return(ret)
    st.write(summary)

    def cal_sma50_sma100_sma200(testDf):
        stocks_SMA= pd.DataFrame()
        for stock in testDf.columns:
            stocks_SMA['{}'.format(stock)] = testDf[stock]
            stocks_SMA["{} SMA50".format(stock)]=testDf[stock].rolling(window = 50).mean()
            stocks_SMA["{} SMA100".format(stock)]=testDf[stock].rolling(window = 100).mean()
            stocks_SMA["{} SMA200".format(stock)]=testDf[stock].rolling(window = 200).mean()
        return stocks_SMA

    moving_avg = cal_sma50_sma100_sma200(df_adj_close)
    st.line_chart(moving_avg)

if st.checkbox('Compare RMSE and R2 Score between different algorithm'):
    # Normalize the Data
    # First thing we need to do is to normalize the data with sklearn's MinMaxScaler function. We created a function for it.
# The data will be scaled between 0 - 1

    def normalize_featuresDF(testDf):

        scaler = MinMaxScaler()
        feature_columns = testDf.columns
        feature_minmax_data = scaler.fit_transform(testDf)
        normalized_features_df = pd.DataFrame(columns=feature_columns, data=feature_minmax_data, index=testDf.index)
        
        
        return normalized_features_df
    
    def split_ValidationSet(features_df, target_df, length=90):
        #need to shift target array because we are prediction n + 1 days price
        target_df = target_df.shift(-1)
        #split validation set . i am spliting 10% latest data for validation.
        #target
        validation_y = target_df[-length:-1]
        validation_x = features_df[-length:-1]
        
        return validation_x, validation_y

    #Now get final_features_df and final_target_df by excluding validation set
    def split_Final_df(normalized_features_df, target_df, v_length=90):
      
        final_features_df = normalized_features_df[:-v_length]
        final_target_df = target_df[:-v_length]
        
        return final_features_df, final_target_df

    #Split final set into training and testing sets
    #splitting training and testing set using sklearn's TimeSeries split
    def split_Train_Test_DF(final_features_df, final_target_df, n_splits=10):
        ts_split = TimeSeriesSplit(n_splits)
        for train_index, test_index in ts_split.split(final_features_df):
            x_train, x_test = final_features_df[:len(train_index)], final_features_df[len(train_index): (len(train_index)+len(test_index))]
            y_train, y_test = final_target_df[:len(train_index)].values.ravel(), final_target_df[len(train_index): (len(train_index)+len(test_index))].values.ravel()
            
        return x_train, y_train, x_test, y_test


        #Method to evaluate the benchmark model and solution model with validate data set
    def model_validateResult(model, model_name):

        model = model(x_train, y_train, validation_x)
        prediction = model.predict(validation_x)
        RMSE_Score = np.sqrt(mean_squared_error(validation_y, prediction))
        R2_Score = r2_score(validation_y, prediction)
        
        #trendline for actual vs prediction

        fig = plt.figure(figsize = (23,10))
        plt.plot(validation_y.index, prediction, color='green', linestyle='dashed', linewidth = 3,
            marker='o', markerfacecolor='green', markersize=8,label = 'Prediction data')
        plt.plot(validation_y.index, validation_y, color='red', linestyle='dashed', linewidth = 3,
            marker='o', markerfacecolor='red', markersize=8,label = 'Actual data')
        plt.plot(figsize = (23,10))
        plt.ylabel('Price',fontsize = 20)
        plt.xlabel('Date',fontsize = 20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.title(model_name + ' Predict vs Actual',fontsize = 20)
        plt.legend(loc='upper left')
        st.pyplot(fig)
        st.write(model_name + ' RMSE: ', RMSE_Score)
        st.write(model_name + ' R2 score: ', R2_Score)
        return RMSE_Score, R2_Score




    #Method to evaluate the final model with testing data set
    def bestModel_validateResult(model, model_name):
    
        #I am giving testing set for the evaluation 
        model = model(x_train, y_train, x_test)
        prediction = model.predict(x_test)
        
        RMSE_Score = np.sqrt(mean_squared_error(y_test, prediction))
        R2_Score = r2_score(y_test, prediction)
        plt.figure(figsize = (23,10))
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.title(model_name + 'Prediction Vs Actual',fontsize = 20)
        plt.plot(y_test, label='test data')
        plt.plot(prediction, label='prediction')
        plt.xlabel('Days',fontsize = 20)
        plt.ylabel('Price',fontsize = 20)
        plt.legend();
        print(model_name + ' RMSE: ', RMSE_Score)
        print(model_name + ' R2 score: ', R2_Score) 
        return RMSE_Score, R2_Score



    def value_Compare(model):
       
        model = model(x_train, y_train, x_test)
        prediction = model.predict(x_test)
        col1 = pd.DataFrame(y_test, columns=['True_value'])
        col2 = pd.DataFrame(prediction, columns = ['Predicted_value'])
        df = pd.concat([col1, col2], axis=1)
        return df
    

    def model_SVR(x_train, y_train, validation_x):
        
        svr_model = SVR(kernel='linear')
        model = svr_model.fit(x_train, y_train)
        return model

    def model_SVRTuning(x_train, y_train, validation_x):
        
        hyperparameters_linearSVR = {
            'C':[0.5, 1.0, 10.0, 50.0, 100.0, 120.0,150.0, 300.0, 500.0,700.0,800.0, 1000.0],
            'epsilon':[0, 0.1, 0.5, 0.7, 0.9],
        }
        
        grid_search_SVR_feat = GridSearchCV(estimator=model_SVR(x_train, y_train, validation_x),
                            param_grid=hyperparameters_linearSVR,
                            cv=TimeSeriesSplit(n_splits=10),
        )

        model = grid_search_SVR_feat.fit(x_train, y_train)
        #print(grid_search_SVR_feat.best_params_)
        
        return model

    def model_Lasso(x_train, y_train, validation_x):
        
        lasso_clf = LassoCV(n_alphas=1000, max_iter=3000, random_state=0)
        model = lasso_clf.fit(x_train,y_train)
    #     prediction = model.predict(validation_x)
        return model


    def model_Ridge(x_train, y_train, validation_x):

        ridge_clf = RidgeCV(gcv_mode='auto')
        model = ridge_clf.fit(x_train,y_train)
    #     prediction = ridge_model.predict(validation_x)
        return model



    #normalizing features df
    normalized_features_df = normalize_featuresDF(testDf)
    target_df = testDf[['Adj Close']]

        #Splitting validation sets from the final features and target df
    validation_x, validation_y = split_ValidationSet(normalized_features_df, target_df)

        #splitting train and test set from validation set
    final_features_df, final_target_df = split_Final_df(normalized_features_df, target_df)

        #splitting train and test
    x_train, y_train, x_test, y_test = split_Train_Test_DF(final_features_df, final_target_df)

    # Getting the RMSE and R2 score by predicting the model.
    
    #SVR model
    RMSE_Score, R2_Score = model_validateResult(model_SVR, model_name = "SVR")

    #SVR model Tuning
    RMSE_Score, R2_Score = model_validateResult(model_SVRTuning, model_name = "SVR_Tuned")

    RMSE_Score, R2_Score = model_validateResult(model_Lasso, model_name = "Lasso")

    RMSE_Score, R2_Score = model_validateResult(model_Ridge, model_name = "Ridge")

    
    #####################################
    def ValidationDataResult(model, model_name):
    
        model = model(x_train, y_train, validation_x)
        prediction = model.predict(validation_x)
        
        RMSE_Score = np.sqrt(mean_squared_error(validation_y, prediction))
        
        R2_Score = r2_score(validation_y, prediction)
        
        model_validation = {model_name:[RMSE_Score,R2_Score]}
        return model_validation


################################################################################


    #Method to evaluate the final model with testing data set
    def TestDataResult(model, model_name):
        
        #I am giving testing set for the evaluation 
        model = model(x_train, y_train, x_test)
        prediction = model.predict(x_test)
        
        RMSE_Score = np.sqrt(mean_squared_error(y_test, prediction))
        
        R2_Score = r2_score(y_test, prediction)
        
        model_validation_test_data = {model_name:[RMSE_Score,R2_Score]}
        
        return model_validation_test_data

    import warnings
    warnings.filterwarnings('ignore')

    model_list = {'SVR': model_SVR,'SVR_Tuning':model_SVRTuning, 'Lasso':model_Lasso,'Ridge':model_Ridge}


    ValidationData_RMSE_R2_Score = []
    TestData_RMSE_R2_Score = []


    for key, value in model_list.items():
        all_model_val = ValidationDataResult(model = value, model_name = key)
        ValidationData_RMSE_R2_Score.append(all_model_val)



    for key, value in model_list.items():
        all_model_val_test = TestDataResult(model = value, model_name = key)
        TestData_RMSE_R2_Score.append(all_model_val_test)
    
    RMSE_ValidationData,R2_Score_ValidationData,models_ValidationData = [],[],[]

    for i in ValidationData_RMSE_R2_Score:
        for key,value in i.items():
            RMSE_ValidationData.append(value[0])
            R2_Score_ValidationData.append(value[1])
            models_ValidationData.append(key)
            


    RMSE_TestData,R2_Score_TestData,models_TestData= [],[],[]

    for i in TestData_RMSE_R2_Score:
        for key,value in i.items():
            RMSE_TestData.append(value[0])
            R2_Score_TestData.append(value[1])
            models_TestData.append(key)

        
        
    Validation_Model_List = pd.DataFrame(np.column_stack([RMSE_ValidationData,R2_Score_ValidationData]), index = models_ValidationData,columns = ['RMSE','R2_Score'] )
    Test_Model_List = pd.DataFrame(np.column_stack([RMSE_TestData,R2_Score_TestData]), index = models_TestData,columns = ['RMSE','R2_Score'] )

    st.write(' ')
    st.write("### Validate Data Set Model Traning result")
    st.write(Validation_Model_List)
    
    st.write(' ')
    st.write("### Test Data Set Model Traning result")
    st.write(Test_Model_List)

    st.write("### Final Result")
    # We select lowest RMSE and Highest R2_Score to select best Model.All results indicate SVR_Tuning is the best model.
    st.write('Min RMSE for Validation DataSet : ', Validation_Model_List['RMSE'].idxmin())
    st.write('\nMax R2_Score for Validation DataSet : ',Validation_Model_List['R2_Score'].idxmax())
    st.write('Min RMSE for Test DataSet: ',Test_Model_List['RMSE'].idxmin())
    st.write('\nMax R2_Score for Test DataSet : ',Test_Model_List['R2_Score'].idxmax())

### --- Sentiment Analysis
title_list = []
time_list = []

# Try to scrap the lastest news block with the headlines
cnbc_news = requests.get('https://www.cnbc.com/world/?region=world').text
soup = BeautifulSoup(cnbc_news, 'lxml')
stock_news_block = soup.find('div', class_="undefined LatestNews-isHomePage LatestNews-isIntlHomepage")
stock_news_list = stock_news_block.find('ul',class_="LatestNews-list").text
headline = stock_news_block.find_all('div', attrs={'class':"LatestNews-headlineWrapper"})

title = stock_news_block.find_all('a', attrs= {'class':"LatestNews-headline"})
lenght = len(title)

for x in title:
    # store in variable
    titles = x.text
    #print(titles)
    title_list.append(titles)

# get time from CNBC lastest news block 
cnbc_news = requests.get('https://www.cnbc.com/world/?region=world').text
soup = BeautifulSoup(cnbc_news, 'lxml')
stock_news_block = soup.find('div', class_="undefined LatestNews-isHomePage LatestNews-isIntlHomepage")
time = stock_news_block.find_all('time', attrs={'class':"LatestNews-timestamp"})
lenght = len(time)

#put news time into a list using for loop
for y in time:
    time = y.text
    time_list.append(time)

# Store data we scrap into a list
data = {
        "Stock Title": title_list,
        "Time": time_list,

    }
df = DataFrame(data, columns=[
        "Stock Title" , "Time"
       
    ])

# preprocessing
df = df.dropna()
df['Stock Title after preprocessing'] = df['Stock Title'].str.lower()
df['Stock Title after preprocessing'] = df['Stock Title after preprocessing'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '' , x))
#remove stopwords
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stop_words])
df['Stock Title after preprocessing']= df['Stock Title after preprocessing'].apply(lambda x: remove_stopwords(x))

# Lemmatization
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])
df['Stock Title after preprocessing'] = df['Stock Title after preprocessing'].apply(lambda text: lemmatize_words(text))

# -------------- Sentiment Analysis -----

from nltk.sentiment import SentimentIntensityAnalyzer
vader = SentimentIntensityAnalyzer()

df['scores'] = df['Stock Title after preprocessing'].apply(lambda review: vader.polarity_scores(review))
df['compound'] = df['scores'].apply(lambda score_dict: score_dict['compound'])
df['sentiment'] = df['compound'].apply(lambda c:'positive' if c>=0 else 'negative')



csv = convert_df(df)

st.write(' ')
if st.checkbox('Show Current CNBC News Title and Sentiment Analysis result to analysis what news will affect stock price'):
    df

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='sentiment_analysis_CBNC_Stock_News.csv',
        mime='text/csv',
    )

st.write('## User Guide')
video_file = open('Userguide.mp4', 'rb')
video_bytes = video_file.read()

st.video(video_bytes)