import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import seaborn as sns
import math 
import streamlit as st 
import numpy as np
import pandas as pd
import altair as alt 
# from streamlit_folium import folium_static
# import folium

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy 
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel 
from PIL import Image

st.set_page_config(page_title="TDS Project",layout='wide')
df_laundry = pd.read_csv("LaundryData.csv")
df_rain = pd.read_csv("Rainfall.csv")

page = st.sidebar.radio(
    'Section Selection',('Group Detail','Data Acquisition / Data Preprocessing', 'EDA', 'ARM / Correlation Matrix',
    'Feature Selection','Classification','Regression','Clustering'))

if page == 'Group Detail':
    st.title("TDS3301 Project")
    
### Question, Data Acquisition, Data Preprocessing
if page == 'Data Acquisition / Data Preprocessing':
    ########## Question ##########
    st.header("Question")
    st.subheader('1. Which is the top washer_no and dryer_no used by the customer?')
    st.subheader('2. When is the most popular period of a day that the customer visited the most?')
    st.subheader('3. Which day in a week has the highest sales based on the number of customers?')
    st.subheader('4. What are the peak hours that customers visit the self-service laundry shop?')
    st.subheader('5. What is the distribution of gender based on customer race?')
    st.subheader('6. What is the distribution of the basket size based on basket colour?')
    st.subheader('7. Which washer and dryer were often used together?')
    st.subheader('8. Did daily rainfall information and day impact the sales of the laundry shop?')
    st.subheader('9. What kind of relationships are there between all the attributes?')
    st.subheader('10. What are the Top-10 and Bottom-10 features based on ‘parts_of_day’ attributes?')
    st.subheader('11. What are the Top-10 and Bottom-10 features based on ‘Basket_Size’ attributes?')
    st.subheader('12. How well performed are the classification models in predicting attributes called ‘parts_of_day’ and “Basket_Size?')
    st.subheader('13. How well performed are the regression models in predicting attributes called ‘Number_Customers’?')
    st.subheader('14. How many clusters are there in the dataset?')

    ########## Data Acquisition ##########
    st.header("Data Acquisition")
    st.info('With the aid of the lecturer, we are provided with a dataset named LaundryData.csv that contains 19 attributes showing customers appearance and behaviour in a self-service coin laundry shop. Luckily, by referring to data.gov.my. (2019) we have also found a supplementary dataset that contains some information about the daily rainfall amount for each state from 2014 until 2020. Below are both dataset information:')
    
    st.subheader("Laundry Dataset")
    st.write(df_laundry)

    st.subheader("Rainfall Dataset")
    st.write(df_rain)

    ########## Data Preprocessing ##########
    st.header("Data Preprocessing")
    st.subheader("Laundry Dataset")

    cateogry = df_laundry.select_dtypes(include=['object']).columns.tolist()

    for i in df_laundry:
        if df_laundry[i].isnull().any():
            if(i in cateogry):
                df_laundry[i] = df_laundry[i].fillna(df_laundry[i].mode()[0])
            else:
                df_laundry[i] = (df_laundry[i].fillna(df_laundry[i].mean())).astype(int)

    df_laundry.isnull().sum() 

    im1a = Image.open("a.PNG")
    st.image(im1a, width = 450,  caption='Laundry Dataset Missing Value and After Data Cleaning')
    st.write(df_laundry)
    st.info('For the data processing procedure, we discovered over 200 rows with missing values. To avoid reducing data, we performed a data cleaning procedure that replaced all missing data with mode values for all object type attributes. While the mean value is used to replace any missing values for integer and float types attributes. The reason for replacing mode values for all object type attributes with missing data is because we are unable to get the mean value. While for the integer and float types attributes replaced with mean values is to avoid an unbalanced range of data.')

cateogry = df_laundry.select_dtypes(include=['object']).columns.tolist()
for i in df_laundry:
    if df_laundry[i].isnull().any():
        if(i in cateogry):
            df_laundry[i] = df_laundry[i].fillna(df_laundry[i].mode()[0])
        else:
            df_laundry[i] = (df_laundry[i].fillna(df_laundry[i].mean())).astype(int)

df_laundry.isnull().sum()

### Exploratory Data Analysis
if page == 'EDA':
    st.header("Exploratory Data Analysis")
    ########## Question 1 ##########
    st.subheader("Question 1: Which is the top washer_no and dryer_no used by the customer?")
    im2a = Image.open("b.PNG")
    st.image(im2a, width = 700,  caption='Bar Graph of Frequency of Washer_No Used by Customers')

    im2b = Image.open("c.PNG")
    st.image(im2b, width = 700,  caption='Bar Graph of Frequency of Dryer_No Used by Customers')
    
    st.info('Figures above show the frequency of Washer_No and Dryer_No used by customers. Results show the Washer_No 3 appeared as the most popular washer used by the customers with 228 users records. While Dryer_No 7 is the most commonly used dryer among all the dryers with 233 users records.')

    ########## Question 2 ##########
    st.subheader("Question 2: When is the most popular period of a day that the customer visited the most?")
    
    df_laundry['Date'] = pd.to_datetime(df_laundry['Date'], format="%d/%m/%Y")
    df_laundry['Time'] =  pd.to_datetime(df_laundry['Time']) - pd.to_datetime(df_laundry['Time']).dt.normalize()
    
    df_laundry['Hour'] = ""
    def get_hour(x):
        seconds = x.seconds
        hours = seconds//3600
        return hours

    for i in range(len(df_laundry['Time'])):
        df_laundry['Hour'][i] = get_hour(df_laundry['Time'][i])

    df_laundry['parts_of_day'] = pd.cut(df_laundry['Hour'], bins = [-1,12,18,24], labels = ['Morning','Afternoon','Evening'])

    im2c = Image.open("d.PNG")
    st.image(im2c, width = 700,  caption='Bar Graph of Frequency of Dryer_No Used by Customers')

    st.info('We have added a new column called ‘parts_of_day’ where we have converted the time of customer arrival from the dataset into different parts of the day as the figure above which are morning (00:00 - 11:59), afternoon (12:00 - 17:59) and evening (18:00 - 23.59). As we can see from the figure above, most of the customers visited the laundry shop during the morning, which is from midnight to the morning.')

    ########## Question 3 ##########
    st.subheader("Question 3: Which day in a week has the highest sales based on the number of customers?")
    df_day = df_laundry.copy()
    df_day = df_day[['Date','Race']]
    df_day = df_day.groupby(['Date']).count()
    df_day = df_day.rename(columns={'Race': 'Number_Customers'})
    df_day['Date'] = df_day.index
    df_day['Date'] = df_day.Date.astype(str)

    list_day = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun','Mon','Thu','Fri',
                'Sun','Mon','Tue','Wed','Thu','Fri','Sun','Mon','Tue','Wed',
                'Fri','Sat','Sun','Wed']

    df_day.insert(2,'Day',list_day)
    df_day.reset_index(drop=True, inplace=True)
    df_day = df_day.rename(columns={'Number_Customers': 'Total_Number_Customers'})

    df_day1 = df_day[['Day','Total_Number_Customers']]
    df_day1 = df_day1.groupby('Day').agg({'Total_Number_Customers': ['sum']})

    df_day1['Days'] = df_day1.index
    df_day1['New_Number_Customers'] =df_day1['Total_Number_Customers']
    df_day1.reset_index(drop=True, inplace=True)
    df_day1.drop(columns=['Total_Number_Customers'],inplace=True)

    bar = alt.Chart(df_day1,title="Number of Customers base on Days").mark_bar().encode(
    x='Days',
    y='New_Number_Customers'
    ).properties(height= 800, width=900)
    st.altair_chart(bar)

    st.info('Due to our lack of skills in programming, we could not convert the date into the day of the week where we manually inserted the day by using python code. The days of the week are separated as Monday, Tuesday, Wednesday, Thursday, Friday, Saturday and Sunday. As we can see from the bar chart above, Saturday and Sunday are having the most customers in a week. We believe that this is because most of the people are not working during the weekend which is why they could use their free time to visit the laundry shop and do their laundry.')

    ########## Question 4 ##########
    st.subheader("Question 4: What are the peak hours that customers visit the self-service laundry shop?")
    df8 = df_laundry.copy()
    df8['Date'] = df8.Date.astype(str)
    df8 = df8.merge(df_day,left_on='Date', right_on='Date')

    q8 = df8.groupby(['Date','Hour']).count()
    q8 = q8.reset_index()

    fig = plt.figure(figsize=(12,8))
    q8b = q8[['Date','Hour','Total_Number_Customers']]
    q8b = q8b.rename(columns={'Total_Number_Customers': 'Number_Customers'})
    
    bar = alt.Chart(q8b,title='Number of Customers base on Days').mark_bar(size=30).encode(
    x='Hour',
    y='Number_Customers'
    ).properties(height= 800, width=900)
    st.altair_chart(bar)

    st.info('Based on the figure, shows the total number of customers visiting the self-service laundry shop every hour from October 2015 to December 2015. Surprisingly, most of the customers visited the laundry shop at 04:00 in the early morning with a record of over 80 customers while 05:00 in the early morning had the fewest customers visiting the laundry shop. There are more customers visiting the laundry shop during the nighttime compared to the late morning as we think that most of the customers work during the daytime.')

    ########## Question 5 ##########
    st.subheader("Question 5: What is the distribution of gender based on customer race?")
    im3 = Image.open("e.PNG")
    st.image(im3, width = 800,  caption='Bar Graph of Distribution of Gender Based on Customer Race')

    st.info('The figure above shows the distribution of gender based on customer race. Indians visited the most frequently compared to other races which recorded about 250 customers within the two months. Chinese and Malay have an approximately equal number of customers visiting the laundry shop with about 230 customers. However, there were less than 100 foreign customers who visited the laundry shop. Overall, the number of male and female customers received a fair distribution across all races but the Malay female customer recorded more visitors than the Malay male customer.')

    ########## Question 6 ##########
    st.subheader("Question 6: What is the distribution of the basket size based on basket colour?")
    im4 = Image.open("f.PNG")
    st.image(im4, width = 800,  caption='Bar Graph of Distribution of Basket Size Based on Customer Basket Colour')

    st.info('The figure above shows the distribution of basket size based on basket colour. In general, a high amount of customers use big baskets and white colour baskets are widely used by the customers. However, the least amount of customers are using the brown big basket and grey small basket which record the lowest number among all kinds of basket. Hence, we can see that it is heavily unbalanced in this case where there is a large amount of customers using big size baskets.')

### Association Rule Mining and Correlation Matrix
if page == 'ARM / Correlation Matrix':
    ########## Association Rule Mining ##########
    st.header("Association Rule Mining")
    st.subheader("Question 7: Which washer and dryer were often used together?")
    
    im5 = Image.open("g.PNG")
    st.image(im5, width = 550,  caption='Figure of Association Rule Mining Results')

    st.info('We used the apriori algorithm to determine which washer and dryer are most often used together by customers. We have set the threshold of minimum support and minimum confidence to 0.1 and 0.3 to determine the association between different washers. The results show that washer number 3 and dryer number 7 received much attention from the customers with a 0.112 support ratio, 0.3947 confidence ratio and 1.3672 lift score. Therefore this result shows a total sense whereby referring to question 1, the result analysed showed the most used washer and dryer by the customer were washer 3 and dryer 7.')

    ########## Correlation Matrix ##########
    st.header("Correlation Matrix")
    st.subheader("Question 8: Did daily rainfall information and day impact the sales of the laundry shop?")
    
    im5c = Image.open("i.PNG")
    st.image(im5c, width = 500,  caption='Correlation Matrix between Rainfall(mm) and Total_Number_Customers')
    
    st.info('We have collected the rainfall dataset of 2015 from data.gov.my. (2019) that provided the rainfall information of all states in Malaysia from 2014 to 2020. Hence, we are able to find the relationship between the sales of the laundry shop and the daily rainfall of Selangor. Firstly, we determine the sales of laundry based on the number of customers of the laundry shop according to the dates provided. By merging both the dataset and the dataframe in question 4 which is total customer based on day, we used a correlation matrix to find the relationship between the attributes found. First, we use the correlation matrix to find the relationship between rainfall per day(mm) and the number of customers per day. Based on the figure above, we can conclude that there is a weak positive relationship between both variables where the rainfall is only having a minor impact on the sales of the laundry shop.')
    
    st.subheader("Question 9: What kind of relationships are there between all the attributes?")
    im5d = Image.open("j.PNG")
    st.image(im5d, width = 1000,  caption='Correlation Matrix Between All the Attribute')

    st.info("After merging the dataset available, we have used Cramer's V correlation matrix to find the relationship between all the attributes in the dataset used. According to Baidu (2008), we have split the ‘rainfall(mm)’ attribute into different rain types such as light rain, moderate rain and heavy rain. The code of correlation matrix was referred from chrisbss1. (2019, September 25) because it is suitable for finding the correlation between the categorical attributes. Based on the figure above, most of the attributes show a very low ratio value of correlation. This concludes that most of the attributes are independent variables which mean there is no relationship between the attributes. However, Kid_Category and With_Kids received the highest correlation score among all the correlations with a 1.0 ratio because Kid_Category can only be determined by With_Kids in this scenario. We can also see that the Rain type and Day are having a correlation score higher than 0.8 with the Number of customers where we believe that both the attributes will affect the number of customers in a day.")

### Feature Selection
if page == 'Feature Selection':
    st.header("Feature Selection")

    ########## parts_of_day ##########
    st.header("Question 10: What are the Top-10 and Bottom-10 features based on ‘parts_of_day’ attributes?")
    
    im6b = Image.open("k2.PNG")
    st.image(im6b, width = 1300,  caption='SNS Plot of Top-10 and Bottom-10 Feature Selection Score for ‘parts_of_day’ attributes?')

    st.info('For the feature selection, we used the package called Boruta to predict a class to maximize performance by removing irrelevant features. We used all the attributes that score above 0.5 in the Boruta feature selection model to proceed with our classification model. Based on the figures Basket_Size, Body_Size, New_Age_Range, Dryer_No, Gender and Spectacles attributes will not be used in the prediction model.')

    ########## Basket_Size ##########
    st.header('Question 11: What are the Top-10 and Bottom-10 features based on ‘Basket_Size’ attributes?')
   
    im6d = Image.open("l2.PNG")
    st.image(im6d, width = 1300,  caption='SNS Plot of Top-10 and Bottom-10 Feature Selection Score for ‘Basket_Size’ attributes?')

    st.info('Again Boruta was used to predict the ‘Basket_Size’ class and all the attributes that score above 0.5 in the model will proceed with our classification model. Based on the figures, Kids_Category, shirt_type, New_Age_Range, Spectacles, pants_type, With_Kids, Gender and Rain Type attributes will not be used in the prediction model.')

### Classification
if page == 'Classification':
    st.header("Classification")

    ########## parts_of_day ##########
    
    st.header("Question 12: How well performed are the classification models in predicting attributes called ‘parts_of_day’ and ‘Basket_Size’?")
    st.info("For the prediction section, we are going to predict two types of classes which are ‘parts_of_day’ and ‘Basket_Size’. First of all, the reason we predict ‘parts_of_day’ attributes is to define what type of customers would like to visit the self-service laundry shop in which part of the day. Next, ‘Basket_Size’ attributes are also predicted in order to find out what size of the basket and what type of customer is likely to bring the big size of the basket to the laundry shop. ")
    
    st.subheader('Random Forest Classifier (parts_of_day)')
    im7 = Image.open("m.PNG")
    st.image(im7, width = 600,  caption='Random Forest Classifier Precision Score for parts_of_day')

    im7b = Image.open("m2.PNG")
    st.image(im7b, width = 600,  caption='Decision Tree Classifier Model Evaluation parts_of_day')

    st.subheader('Decision Tree Classifier (parts_of_day)')
    im7c = Image.open("m3.PNG")
    st.image(im7c, width = 600,  caption='Random Forest Classifier Precision Score for parts_of_day')

    im7d = Image.open("m4.PNG")
    st.image(im7d, width = 600,  caption='Decision Tree Classifier Model Evaluation for parts_of_day')

    st.info("We employed two distinct classifiers for the 'parts of day' attribute prediction which are random forest (RF) and decision tree (DT) classifiers. As we can see, the DT classifier outperformed the RF classifier in terms of accuracy and consistency at different depths. The DT classifier received a peak accuracy of 0.8 and a constant accuracy of about 0.75-0.8, whereas the RF classifier had a peak accuracy of 0.72 and a steady accuracy of around 0.6-0.72. Hence, DT is better at predicting the class.")

    ########## Basket_Size ##########
    st.subheader('Naive Bayes (Basket_Size)')
    im7e = Image.open("m5.PNG")
    st.image(im7e, width = 600,  caption='Naive Bayes Precision Score for Basket_Size')

    st.subheader('K Nearest Neighbour Classifier (Basket_Size)')
    im7f = Image.open("m6.PNG")
    st.image(im7f, width = 600,  caption='K Nearest Neighbour Model Evaluation for Basket_Size')

    st.subheader('Naive Bayes and K Nearest Neighbour Classifier ROC Curve (Basket_Size)')
    im7g = Image.open("m7.PNG")
    st.image(im7g, width = 600,  caption='Decision Tree and K Nearest Neighbour Classifier ROC Curve for Basket_Size')
    
    st.info("The findings in question 6 shows the basket size is heavily unbalanced where there is a large amount of big size basket used by the customer. Hence, an oversampling method has been used to balance out the distribution of big and small baskets. Naive Bayes (NB) and K Nearest Neighbour (KNN) classifiers were used to predict the 'Basket Size' attribute. The KNN classifier outperformed the NB classifier in terms of accuracy, with an accuracy ratio of 0.85 for the KNN classifier and 0.73 for the NB classifier. We are unable to compare the model evaluation section since the NB classifier is unable to do tuning. However, we did calculate the AUC and plotted the ROC Curve for both classifiers to compare the classifiers. To determine whether classifier performance is superior, we must first determine which classifier curves are higher or closer to 1. As we can see KNN classifier ROC curve and AUC score is somewhat higher than the NB classifier, where the KNN classifier AUC score is 0.79 and the NB classifier is 0.78. Thus, KNN is better at predicting the class.")

### Regression
if page == 'Regression':
    st.header('Regression')
    st.subheader('Question 13: How well performed are the regression models in predicting attributes called ‘Number_Customers’?')
    st.info("In the regression section, we are going to use the three different regression models which are RF, DT and Lasso Regression to do prediction on the attribute called ‘Number_Customers’. The reason for doing the regression process is to predict the number of customers by every hour of a day.")
    
    im8 = Image.open("n.PNG")
    st.image(im8, width = 1300,  caption='First Previous 14 hours of the Number of Customers')

    im8 = Image.open("n2.PNG")
    st.image(im8, width = 1300,  caption='Regression Models Evaluation')

    st.info("We have used the previous 24 hours of the number of customers as the features of our regression model to predict the upcoming number of customers visiting the laundry shop. The regressor used were RF, DT and Lasso Regression where each of them respectively scored a mean squared error with 2.22, 3.45, 6.47. As we can see from the figure above, RF and DT regression predicted values were roughly the same which is why the mean squared error from both the models are similar. Lasso regression performed the worst out of the 3 regression models here which had a significantly high mean squared error compared with another 2 models. It also has the worst result on a graph where the predicted values it gets is having a high similarity with the actual values.")

if page == 'Clustering':
    st.header('Clustering')
    st.subheader("Question 14: How many clusters are there in the dataset?")
    im9 = Image.open("p.PNG")
    st.image(im9, width = 1300,  caption='Clustering Chart')

    st.info("As for the clustering technique, we have used K-Prototype to find the clusters available in our dataset. Due to the dataset available having different types of variables such as numerical and categorical, K-means and K-modes were not suitable for clustering the dataset. Hence, K-Prototype was used on clustering the variables because it is able to deal with categorical and numerical data at the same time. As we can see from the scatter plot above, the elbow criterion occurred at the number cluster of 3. Hence, we can conclude there are 3 different clusters in total for the attributes used.")