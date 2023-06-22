# Stock-Market-Price-Prediction-using-LSTM

Please refer to the ppt slides and report doc for deep explaination. 

For the analysts and researchers, predicting the stock prices every year always has been challenging. No doubt that the stock market community specially the investors are interested to do the research in area of stock market price prediction. Many investors are keen to know about the future prices of the shares and stocks for a good investment. Deep learning neural networks help the analysts to know about the situation of the stock market for the future. In this project, we are implementing the LSTM on the data-set in order to predict the stock-market prices for the future.
 

Abstract— For the analysts and researchers, predicting the stock prices every year always has been challenging. No doubt that the stock market community specially the investors are interested to do the research in area of stock market price prediction. Many investors are keen to know about the future prices of the shares and stocks for a good investment. Deep learning neural networks help the analysts to know about the situation of the stock market for the future. In this project, we are implementing the LSTM on the data-set in order to predict the stock-market prices for the future. 
Keywords— LSTM, Stock Price Prediction, RMSE, Neural Network, Artificial Neural Network, Stock Market, RNN

I.	INTRODUCTION 
The stock market is a gathering of buyers and sellers who express their interest in trading stocks that are issued by businesses to raise funds and are purchased by investors to acquire a stake in the business. The stock market is constantly volatile, and it is impossible to forecast what will happen next because the stock prices of firms are constantly changing. As is common knowledge, a company's stock is influenced by a variety of variables, taking into account the nonlinearities and discontinuities of the variables thought to have an impact on stock markets. The reasons include things like news about businesses, political events, natural disasters, etc. One of the most crucial topics for academic and financial research is stock price prediction.
The stock market is a dynamic, complicated, and evolutionary system. Noise, data intensity, non-stationarity, uncertainty, and hidden relationships are characteristics of market prediction. A difficult and crucial study area has been the forecasting of market trends. The data is difficult since it is erratic and noisy. It is significant because it has the potential to influence key decisions. Companies often invest large sums of money and trade their shares in stock markets.
The idea of employing the LSTM algorithm to make an effort at stock market prediction on the Microsoft dataset is explored in the report that follows. This is a crucial piece of knowledge since it establishes a connection between the fields of computer science and finance and can lead to several future prediction approaches. A type of RNN known as LSTM can learn long sequences, and the algorithms it uses to do so are known as long-short term memories.
The capacity of the Long-Short Term Memory algorithm to forecast stock market movements for the Microsoft dataset will be examined in this research. We will explain each and every stage of LSTM implementation and will compare the outputs on different hyperparameters and algorithms. We'll take a number of stages as we try to explore and respond to the research topic.
II.	LITERATURE VIEW
The Efficient Market Hypothesis, which contends that it is difficult to predict the market because it is efficient, has been refuted by stock market prediction. It has been demonstrated by researchers that it is possible to forecast the stock market. Making future stock market predictions is a crucial skill for investors who want to succeed financially. Additionally, it aids investors in choosing whether to sell or acquire in order to make more money.
In M.Suresh Babu et al., 2012 [1], this paper investigates the important clustering calculations: K-Means, Hierarchical grouping calculation, and reverse K-Means and examines the execution of these noteworthy clustering calculations on the part of successfully class savvy group building capacity of calculation. Three stages make up the suggested strategy. They convert each financial report into an element vector first, then group the converted element vectors into clusters using several levelled agglomerative grouping techniques. In monetary reports, they take into account both qualitative and quantitative highlights. Second, they combine the benefits of two grouping strategies to suggest a strong clustering strategy. Third, choosing an appropriate number of parts in HAC can reduce the bunches generated and, in this way, improve the K-means clustering's ability to produce groups that are of a particular type.
According to Mahajan Shubhrata D et al., 2016 [2], the purpose of this work is to forecast future stock value. Parse Records will then calculate the predicted value and transmit it to the client. Additionally, use the automation concept to perform tasks like buying and selling shares. Use the Naive Bayes algorithm for that. Downloading log shapes from the Hurray Back site and storing them in a dataset allows for real-time access. The investigations show that Naive Bayes Algorithm has a great capacity for predicting the appearance of interest in the offer market.
By predicting a stock's earnings using a kind of rigorous machine learning computations known as ensemble learning, Luckyson Khaidem et al. (2016) [3] proposed a revolutionary way to reduce the risk of interest in stock markets. Logistic Regression, Gaussian Discriminant Analysis, Quadratic Discriminant Analysis, and SVM are the four administered learning computations they used.
Shraddha Varma, Harshal Patel, and Murtaza Roondiwala 2018 [4] worked on the study that used historical data for training and validation in order to predict stock market prices using LSTM. Their model was founded on the NIFTY, which was predicting stocks.

III.	PROBLEM STATEMENT
The stock market is an evolutionary, complex and a dynamic system. Market prediction is characterized by noise, data intensity, non-stationary, uncertainty and hidden relationships. The prediction of trend in stock market exchange has been a challenging and important research topic. It is challenging because the data is noisy and not stationary. It is important because it can yield important results for decision makers.

IV.	PROPOSED METHODOLOGY
The following report will delve into the concept of using LSTM algorithm to attempt to predict stock market prediction. This is an important aspect to knowledge because it can give rise to many prediction techniques in the future and it introduces a link between the ﬁnance ﬁeld and the ﬁeld of computer science. 

Python3 using Jupyter Notebook The interactive notebooks used in this course have an "ipynb" file extension. The Jupyter Notebook has replaced the iPython Notebook. It is an interactive computing environment where code execution, rich text, mathematics, graphs.

A.	Data Collection
Microsoft dataset is collected from the Kaggle and there are various columns in the dataset for Microsoft MSFT .csv file [5] having the columns named as date, open, high, low, last, close. The starting and final price at which the stock is traded on a particular day is represented by the Open and Close column in the dataset. On the other hand, the columns such as High, Low and Last indicate the maximum, minimum, and last price per share for the day. The dataset does not contain any null and missing values.

Dataset in the file is divided into three parts such as 75% for the training purposes while 10% refer to the validation set (taking last 90 rows of data to be validation set) whereas 15% data is allocated for testing.
 

Also, there is a correlation analysis between the columns of the dataset which is show as below

B.	Data Processing
The dataset is processed by the transformation or normalization process. It gets cleaned if there is null or missing value in the data – in our case we do not have missing values and null is false then the the data in the files is integrated. We pass the impacted columns and data to our neural network. 

 
The columns that are significant such as Date, open, high, low, close, and volume from the dataset are chosen.

C.	Model Development
In this section, we'll go over our system's methodology. Our system is divided into the following stages: -

 

(Fig: LSTM Stages)

For problems involving sequence prediction, LSTMs are frequently employed and have shown to be incredibly successful. They function so well because LSTM can remember past knowledge that is significant while forgetting less-important information. Three gates make up an LSTM:

o	The input gate: The input gate updates the cell state with additional data.
o	The forget gate: It eliminates data that the model no longer needs.
o	The output gate:: The information to be shown as output is chosen by the LSTM's output gate.

The processed data is then passed into the neural network and the model is trained with random weights and hyperparameters. The model consists of two neural layers and the two dense layers with the activation functions ReLU. The model is trained over below hyperparameters:

epochs=200, batch_size=8, verbose=1

The implementation of LSTM as follow:

Where the LSTM having two layers having 100 units each and dense layer is defined to be as 25 and 1.

V.	OUTPUT AND ANALYSIS 

A.	Visualizations of the Outcomes
The following graph shows the relation between the actual and predicted stock price:
Also, the results that show the dataset from training and find the prediction of the test data is as below:

 
B.	Analyzing the Results
To check the performance of the model, rms is calculated. Root Mean Square Error is the square of the mean of difference (y_test – y_predictions). 
 
The results are logical and reasonable since they are stable, robust, and have a low error ratio. RMSE is discovered to be under:

 

It may vary depend upon the size of the data and number of epochs on which we trained our model. However, the above plot defines the close prediction of output and checks the accuracy of LSTM model compared to the true data. In addition, the score results of our LSTM trained model using trained and test data came out to be:

 
The model summary is given as below:

 
C.	Results on Different Epochs and Parameters
The observations are carried on increasing the number epochs and the results noted down are below:

Parameter	No. of Epochs	Root Mean Square Error (RMSE)
Open/Close	200	1.81
Open/ Close	500	0.53
Open/ Close/High/Low	200	0.1004
Open/ Close/High/Low	500	0.0989

(Table 1: Comparison on changing epoch and parameters)

D.	Comparison with other Agorithms 
The results are carried out using the Decision Tree Regressor and the output is unsatisfactory that can be seen below:

 
(Fig 6: Results from Decision Tree Regressor)

While implementing the stock prices prediction using K-NN, we can see the prediction in the visualization which is not as accurate that LSTM provide us the result 

Here is a comparison of our LSTM with a deep and non-deep learning model and it can be observed that LSTM model give us the better performance and predictions compared to K-NN and Decision Tree.

E.	Compariosn based on different Dataset

The LSTM model has been tested on Apple dataset [5] and the prediction came out to be below for (500 Epochs and 1 batch size), the visualization is similar and the trend is quite match with the original data.


F.	Tuning the Hyperparameters with the Comparison

On setting the one layers to LSTM with 16 units and one dense layer, results are not that much attractive compared to the LSTM having two layers with 100 units and dense layer set to 25 and 1 respectively.  The plot for LSTM having two layers with 100 units and dense layer set to 25 and 1 [7] is shown below:
 

On running the model for the 200 Epoch and batch size 8, the results were very poor but on increasing the epochs and decreasing the batch size, the results got better. The visualizations can be seen below on different hypermeters for Apple dataset.


VI.	CONCLUSION
In this work, we implemented the LSTM on Microsoft Price Prediction dataset and carried out quite promising results. Overall, the accuracy of LSTM is satisfactory and the model is predicting the future close prices with good score and better rms error. In this work, the well-known deep learning algorithm R-NN which provides precise results is used. It will be helpful for the investors and will provide a brief knowledge who want to analyze and predict the stock market prices for the future.

VII.	DISCUSSION AND FUTURE WORK
The results can be polished more and the error can be reduced if we train our model to large dataset and increase the epoch as by tuning the hyperparameters, the model results will be more accurate. No model is 99% accurate but we can experiment it with various activation functions and can alter the model structure to get the better results.

In the future, we can extend to other application and can perform sentiment analysis on knowing about the people thinking regarding the market prices. In addition, we can add Facebook and Twitter data in our algorithm collected based on people opinions and predictions to better understand how the market feels about price changes for specific shares.

ACKNOWLEDGMENT 
The project is implemented under the supervision of the lecturer while many senior and colleagues helped and contributed in the deep learning report and analysis. It has been reviewed thoroughly and changes have been made where required. The professor helped us out in many phases while working on the project. Thanks to the people who provided support in writing the research report and contributed in the project implementation.












REFERENCES

[1]   C. N. Babu and B. E. Reddy, “Selected Indian stock predictions using a hybrid ARIMA-GARCH model,” 2014 Int. Conf. Adv. Electron. Comput. Commun. ICAECC 2014, 2015.
[2]	M. D. Shubhrata, D. Kaveri, T. Pranit, and S. Bhavana, “Stock Market Prediction and Analysis Using Naïve Bayes,” Int. J. Recent Innov. Trends Comput. Commun., vol. 4, no.11, pp. 121–124, 2016.
[3]	L. Khaidem, S. Saha, and S. R. Dey, “Predicting the direction of stock market prices using random forest,” vol. 0, no. 0, pp. 1– 20, 2016.
[4]	Murtaza Roondiwala Harshal Patel Shraddha Varma Predicting ― Stock Prices Using         LSTM: International Journal of Science and Research (IJSR) ISSN (Online): 2319-7064 Index Copernicus Value (2015): 78.96 | Impact Factor (2015): 6.391.
[5]	https://www.kaggle.com/datasets/hemamounika/msft-dataset 
[6]	https://www.kaggle.com/datasets/varpit94/apple-stock-data-updated-till-22jun2021
[7]	https://www.kaggle.com/shehreenmushtaq/stockpriceusinglstm/




