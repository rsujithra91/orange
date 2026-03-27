from pandas import read_csv
# define the location of the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer.csv"
# load the dataset
dataset = read_csv(url, header=None)
 
print(dataset.head())
         0          1        2      3      4    5        6           7      8  \
0  '40-49'  'premeno'  '15-19'  '0-2'  'yes'  '3'  'right'   'left_up'   'no'   
1  '50-59'     'ge40'  '15-19'  '0-2'   'no'  '1'  'right'   'central'   'no'   
2  '50-59'     'ge40'  '35-39'  '0-2'   'no'  '2'   'left'  'left_low'   'no'   
3  '40-49'  'premeno'  '35-39'  '0-2'  'yes'  '3'  'right'  'left_low'  'yes'   
4  '40-49'  'premeno'  '30-34'  '3-5'  'yes'  '2'   'left'  'right_up'   'no'   

                        9  
0     'recurrence-events'  
1  'no-recurrence-events'  
2     'recurrence-events'  
3  'no-recurrence-events'  
4     'recurrence-events'  
print(dataset.head())
print(dataset.info())
print(dataset.describe())
         0          1        2      3      4    5        6           7      8  \
0  '40-49'  'premeno'  '15-19'  '0-2'  'yes'  '3'  'right'   'left_up'   'no'   
1  '50-59'     'ge40'  '15-19'  '0-2'   'no'  '1'  'right'   'central'   'no'   
2  '50-59'     'ge40'  '35-39'  '0-2'   'no'  '2'   'left'  'left_low'   'no'   
3  '40-49'  'premeno'  '35-39'  '0-2'  'yes'  '3'  'right'  'left_low'  'yes'   
4  '40-49'  'premeno'  '30-34'  '3-5'  'yes'  '2'   'left'  'right_up'   'no'   

                        9  
0     'recurrence-events'  
1  'no-recurrence-events'  
2     'recurrence-events'  
3  'no-recurrence-events'  
4     'recurrence-events'  
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 286 entries, 0 to 285
Data columns (total 10 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   0       286 non-null    object
 1   1       286 non-null    object
 2   2       286 non-null    object
 3   3       286 non-null    object
 4   4       278 non-null    object
 5   5       286 non-null    object
 6   6       286 non-null    object
 7   7       285 non-null    object
 8   8       286 non-null    object
 9   9       286 non-null    object
dtypes: object(10)
memory usage: 22.5+ KB
None
              0          1        2      3     4    5       6           7  \
count       286        286      286    286   278  286     286         285   
unique        6          3       11      7     2    3       2           5   
top     '50-59'  'premeno'  '30-34'  '0-2'  'no'  '2'  'left'  'left_low'   
freq         96        150       60    213   222  130     152         110   

           8                       9  
count    286                     286  
unique     2                       2  
top     'no'  'no-recurrence-events'  
freq     218                     201  
# retrieve the array of data
data = dataset.values
# separate into input and output columns
X = data[:, :-1].astype(str)
y = data[:, -1].astype(str)
# summarize
print('Input', X.shape)
print('Output', y.shape)
Input (286, 9)
Output (286,)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
# ordinal encode input variables
ordinal_encoder = OrdinalEncoder()
X = ordinal_encoder.fit_transform(X)
# ordinal encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
# summarize the transformed data
print('Input', X.shape)
print(X[:5, :])
print('Output', y.shape)
print(y[:5])
Input (286, 9)
[[2. 2. 2. 0. 1. 2. 1. 2. 0.]
 [3. 0. 2. 0. 0. 0. 1. 0. 0.]
 [3. 0. 6. 0. 0. 1. 0. 1. 0.]
 [2. 2. 6. 0. 1. 2. 1. 1. 1.]
 [2. 2. 5. 4. 1. 1. 0. 4. 0.]]
Output (286,)
[1 0 1 0 1]
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
# split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# ordinal encode input variables
ordinal_encoder = OrdinalEncoder()
ordinal_encoder.fit(X_train)
X_train = ordinal_encoder.transform(X_train)
X_test = ordinal_encoder.transform(X_test)
# ordinal encode target variable
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)
# define the model
model = LogisticRegression()
# fit on the training set
model.fit(X_train, y_train)
# predict on test set
yhat = model.predict(X_test)
# evaluate predictions
accuracy = accuracy_score(y_test, yhat)
print('Accuracy: %.2f' % (accuracy*100))
