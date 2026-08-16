import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

DATA_TRAIN = pd.read_csv('../dataset/train.csv')
DATA_TEST = pd.read_csv('../dataset/test.csv')

TITLE = ['Mr.', 'Miss.', 'Mrs.', 'Master.', 
         'Dr.', 'Rev.', 'Major.', 'Col.', 'Capt.',
         'Mlle.', 'Mme.', 'Ms.', 'Don.', 'Lady.', 'Sir.',
         'Countess.', 'Jonkheer.', 'Dona.'] 

def encode_sex(data):
    data.Sex = data.Sex.replace('male', 0)
    data.Sex = data.Sex.replace('female', 1)

    return data

def name_to_title(data):
    data['Title'] = ""
    
    i = 0
    while i < len(data):
        for t in TITLE:
            if t in data['Name'][i]:
                if t == "Mlle.":
                    data.loc[i, 'Title'] = "Miss."
                    break
                elif t == "Mme.":
                    data.loc[i, 'Title'] = "Mrs."
                    break
                else:
                    data.loc[i, 'Title'] = t
                    break
            else:
                data.loc[i, 'Title'] = "Unknow"
                
        i = i + 1

    return data

def input_age_according_to_title(data):

    data["age_was_missing"] = data.Age.isna()

    avg_age = (data.groupby('Title')['Age'].mean())
    
    i = 0
    while i < len(data):
        
        if np.isnan(data['Age'][i]):
            data.loc[i, 'Age'] = avg_age[data.loc[i, 'Title']]

        i = i + 1

    return data, avg_age

def is_familly(data):
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    return data

def compute_features(data, features_list):
    y = data.Survived
    features = features_list
    X = data[features]

    return X, y

def prepare_my_data(data, features_list):
    standardize_data = encode_sex(data)
    standardize_data = is_familly(standardize_data)
    standardize_data = name_to_title(standardize_data)
    standardize_data, avg_age = input_age_according_to_title(standardize_data)
    X_standard, y_standard = compute_features(standardize_data, features_list)
    
    return X_standard, y_standard, avg_age

def input_age_according_to_title_full(data, avg_age):

    i = 0
    while i < len(data):
        
        if np.isnan(data['Age'][i]):
            if np.isnan(avg_age[data.loc[i, 'Title']]):
                data.loc[i, 'Age'] = 28
            else:
                data.loc[i, 'Age'] = avg_age[data.loc[i, 'Title']]

        i = i + 1

    return data

def compute_features_full(data, features_list):
    features = features_list
    X = data[features]

    return X

def prepare_my_data_full(data, features_list, avg_age):
    standardize_data = encode_sex(data)
    standardize_data = is_familly(standardize_data)
    standardize_data = name_to_title(standardize_data)
    standardize_data = input_age_according_to_title_full(standardize_data, avg_age)
    X_standard = compute_features_full(standardize_data, features_list)
    
    return X_standard, standardize_data

def train_full(X,y):
    titanic_model_full = RandomForestClassifier(random_state=1, max_depth=5, min_samples_leaf=5)
    titanic_model_full = titanic_model_full.fit(X, y)
    return titanic_model_full

def predict_test(titanic_model_full, X, data_test):
    titanic_preds = titanic_model_full.predict(X)
    
    output = pd.DataFrame({'PassengerId': data_test.PassengerId,
                        'Survived': titanic_preds})
    output.to_csv('../submission/submission_V1.csv', index=False)

if __name__ == "__main__":
    X_familly, y_familly, avg_age = prepare_my_data(DATA_TRAIN, ['Sex', 'Pclass', 'Age', 'FamilySize'])
    X, standardize_data = prepare_my_data_full(DATA_TEST, ['Sex', 'Pclass', 'Age', 'FamilySize'], avg_age)
    titanic_model_full = train_full(X_familly, y_familly)
    predict_test(titanic_model_full, X, standardize_data)