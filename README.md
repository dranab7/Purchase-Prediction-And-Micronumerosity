# Purchase-Prediction-And-Micronumerosity

Project Overview
This project focuses on predicting customer purchase behavior using a dataset with 50 samples. The dataset includes demographic and review information about customers, with the target variable being whether the customer made a purchase. The project employs a Random Forest Classifier to build the prediction model and evaluates its performance on a test set.

Step-by-Step Implementation
Importing Libraries

python
Copy code
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
Essential libraries for data manipulation, model building, and evaluation are imported.

Loading the Data

python
Copy code
purchase = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/Customer%20Purchase.csv')
purchase.head()
The dataset is loaded from a URL and the first few rows are displayed to understand the structure and content of the data.

Data Inspection

python
Copy code
purchase.info()
purchase.describe()
The dataset is inspected to check for data types, missing values, and to get summary statistics. This reveals that there are no missing values and provides insight into the distribution of numerical features.

Defining Target and Features

python
Copy code
y = purchase['Purchased']
X = purchase.drop(['Purchased','Customer ID'],axis=1)
The target variable (y) is set to Purchased, and the feature set (X) includes Age, Gender, Education, and Review. Customer ID is dropped as it is not a relevant feature for prediction.

Encoding Categorical Variables

python
Copy code
X.replace({'Review':{'Poor':0,'Average':1,'Good':2}},inplace=True)
X.replace({'Education':{'School':0,'UG':1,'PG':2}},inplace=True)
X.replace({'Gender':{'Male': 0,'Female':1}},inplace=True)
X.head()
Categorical variables are encoded into numerical values for model compatibility.

Splitting the Data

python
Copy code
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=2529)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
The dataset is split into training and testing sets, with 80% of the data used for training and 20% for testing.

Model Selection and Training

python
Copy code
model = RandomForestClassifier()
model.fit(X_train, y_train)
A Random Forest Classifier is selected and trained on the training data.

Making Predictions

python
Copy code
y_pred = model.predict(X_test)
y_pred
Predictions are made on the test data.

Model Evaluation

python
Copy code
confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
The model's performance is evaluated using a confusion matrix, accuracy score, and classification report. These metrics provide insights into the model's effectiveness.

Results and Analysis
Confusion Matrix:

python
Copy code
array([[2, 1],
       [4, 3]])
The confusion matrix indicates that out of 10 test samples, the model correctly predicted 2 'No' purchases and 3 'Yes' purchases. However, it also misclassified 1 'No' as 'Yes' and 4 'Yes' as 'No'.

Accuracy Score:

python
Copy code
0.5
The model achieved an accuracy of 50%, indicating it correctly predicted the purchase behavior for half of the test samples.

Classification Report:

python
Copy code
              precision    recall  f1-score   support

          No       0.33      0.67      0.44         3
         Yes       0.75      0.43      0.55         7

    accuracy                           0.50        10
   macro avg       0.54      0.55      0.49        10
weighted avg       0.62      0.50      0.52        10
Precision and Recall: For 'No' purchases, the model has a precision of 0.33 and a recall of 0.67, indicating it is more likely to miss 'No' purchases but when it does predict 'No', it is less often correct. For 'Yes' purchases, the precision is higher at 0.75, but recall is lower at 0.43, indicating a higher rate of false negatives.
F1-Score: The F1-score for 'No' is 0.44, and for 'Yes' it is 0.55, suggesting a balanced trade-off between precision and recall.
Overall Performance: The macro average F1-score is 0.49, and the weighted average is 0.52, reflecting moderate model performance given the small dataset size.
Conclusion
This project highlights the challenges of predicting customer purchase behavior with a small dataset (micro-numerosity). The Random Forest Classifier achieved moderate accuracy, but performance could be improved with a larger dataset and more feature engineering. Future steps could include exploring other classification algorithms, addressing class imbalance, and incorporating additional features to enhance model accuracy and robustness.
