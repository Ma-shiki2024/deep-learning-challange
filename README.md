Deep Learning Challenge



Background  

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

EIN and NAME—Identification columns

APPLICATION_TYPE—Alphabet Soup application type

AFFILIATION—Affiliated sector of industry

CLASSIFICATION—Government organization classification

USE_CASE—Use case for funding

ORGANIZATION—Organization type

STATUS—Active status

INCOME_AMT—Income classification

SPECIAL_CONSIDERATIONS—Special considerations for application

ASK_AMT—Funding amount requested

IS_SUCCESSFUL—Was the money used effectively

Step 1: Preprocess the Data

Step 2: Compile, Train, and Evaluate the Model

Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

Step 3: Optimize the Model

Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%. Use any or all of the following methods to optimize your model:

Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as: Dropping more or fewer columns.

Creating more bins for rare occurrences in columns.

Increasing or decreasing the number of values for each bin.

Add more neurons to a hidden layer.

Add more hidden layers.

Use different activation functions for the hidden layers.

Add or reduce the number of epochs to the training regimen.

Step 4: Write a Report on the Neural Network Model
For this part of the assignment, you’ll write a report on the performance of the deep learning model you created for Alphabet Soup. The report should contain the following:

Overview of the analysis: Explain the purpose of this analysis.
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.
Given the dataset we are using machine learning models to train and predict the success outcomes for potential funding oppurtunities .
Based on the training data , Model should be able predict which applicants for funding have a higher success rate and hence Alphabet soup can invest in these ventures .




Results: Using bulleted lists and images to support your answers, address the following questions:

Data Preprocessing

  What variable(s) are the target(s) for your model?

  IS_SUCCESSFUL variable is the target for our model

  What variable(s) are the features for your model?
  
  Following are the features for my model

  Index(['STATUS', 'ASK_AMT', 'IS_SUCCESSFUL', 'APPLICATION_TYPE_Other_app_type',
       'APPLICATION_TYPE_T19', 'APPLICATION_TYPE_T3', 'APPLICATION_TYPE_T4',
       'APPLICATION_TYPE_T5', 'APPLICATION_TYPE_T6', 'APPLICATION_TYPE_T7',
       'APPLICATION_TYPE_T8', 'AFFILIATION_CompanySponsored',
       'AFFILIATION_Family/Parent', 'AFFILIATION_Independent',
       'AFFILIATION_National', 'AFFILIATION_Other', 'AFFILIATION_Regional',
       'CLASSIFICATION_C1000', 'CLASSIFICATION_C1200', 'CLASSIFICATION_C2000',
       'CLASSIFICATION_C2100', 'CLASSIFICATION_C3000', 'CLASSIFICATION_Other',
       'USE_CASE_CommunityServ', 'USE_CASE_Heathcare', 'USE_CASE_Other',
       'USE_CASE_Preservation', 'USE_CASE_ProductDev',
       'ORGANIZATION_Association', 'ORGANIZATION_Co-operative',
       'ORGANIZATION_Corporation', 'ORGANIZATION_Trust', 'INCOME_AMT_0',
       'INCOME_AMT_1-9999', 'INCOME_AMT_10000-24999',
       'INCOME_AMT_100000-499999', 'INCOME_AMT_10M-50M', 'INCOME_AMT_1M-5M',
       'INCOME_AMT_25000-99999', 'INCOME_AMT_50M+', 'INCOME_AMT_5M-10M',
       'SPECIAL_CONSIDERATIONS_N', 'SPECIAL_CONSIDERATIONS_Y'],
      dtype='object')

  
  What variable(s) should be removed from the input data because they are neither targets nor features?

  In addition to name and EIN , I have removed STATUS', 'SPECIAL_CONSIDERATIONS_N','SPECIAL_CONSIDERATIONS_Y variables as these were neither targets nor features 


  Compiling, Training, and Evaluating the Model



 How many neurons, layers, and activation functions did you select for your neural network model, and why?

 I set up three layers and three diffrent activation functions for each of those layers .
 My activation functions used were "relu" ,"tanh" and "sigmoid" .
 I chose these parameters to get higher accuracy of predictions 
 
 Were you able to achieve the target model performance?
 I was able to get a accuracy for 72% of the predictions with 2 hidden layers and all inclusive feature set 

 What steps did you take in your attempts to increase model performance?

I took the following steps to improve the model performance 
* i dropped these columns for my feature set 
'STATUS', 'SPECIAL_CONSIDERATIONS_N','SPECIAL_CONSIDERATIONS_Y'
I changed the bin values for application type and classification type based on the cut off value .
I increased/decreased the cut off value in the training set and ran multiple iterations of the model 
I added an additional hidden layer to make it three and used the activation function of "Sigmoid" to increase the accuracy .
I also used RandomForestClassifier model in addition to the neural network model as RandomForestClassifier are better fit for solving classification problems .
Increase the EPOCHS for the training regimen from 20 to 50 to 100 and trained the model multiple times 


Summary: Overall , Having tweaked parameters of the model , the accuracy of the prediction ranged between 72% to 75 %
further analysis can be perfoemed on the dataset to evaluate columns that have more weightage on the success of the venture .
However , Since the success of the venture is depended on lot of other factors including demographic , market conditions etc 
There is potential to add and capture more columns and determing the outcome of our venture 
So , the model can be more accurate .
We could also try different models like k-Nearest Neighbors (k-NN),Decision Trees,Support Vector Machines (SVM), Logistic Regression and analyse the accuracy of the model .

