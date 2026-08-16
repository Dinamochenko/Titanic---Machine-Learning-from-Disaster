import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

DATA_TRAIN = pd.read_csv('../dataset/train.csv')
DATA_TEST = pd.read_csv('../dataset/test.csv')

def encode_sex(data):
    data.Sex = data.Sex.replace('male', 0)
    data.Sex = data.Sex.replace('female', 1)

    return data

def prepare_data(data):
    y = data.Survived
    features = ['Sex', 'Pclass']
    X = data[features]

    return y, X

def split_test(y, X):
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
    return train_X, val_X, train_y, val_y

def train_model(train_X, train_y):
    titanic_model = RandomForestClassifier(random_state=0)
    titanic_model = titanic_model.fit(train_X, train_y)
    return titanic_model
    
def titanic_preds(titanic_model, val_X, val_y):
    titanic_p = titanic_model.predict(val_X)
    print(f"The accuracy of the model is: {round(accuracy_score(val_y, titanic_p) * 100, 2)} %")

def train_full_model(y,X):
    titanic_model_full = RandomForestClassifier()
    titanic_model_full = titanic_model_full.fit(X, y)
    return titanic_model_full

def predict_test(titanic_model_full, X, data_test):
    titanic_preds = titanic_model_full.predict(X)
    output = pd.DataFrame({'PassengerId': data_test.PassengerId,
                        'Survived': titanic_preds})
    output.to_csv('../submission/submission_V0.csv', index=False)

def prepare_data_test(data):
    features = ['Sex', 'Pclass']
    X = data[features]

    return X

if __name__ == "__main__":
    data = encode_sex(DATA_TRAIN)
    y, X = prepare_data(data)

    # Accuracy of the model on training data
    train_X, val_X, train_y, val_y = split_test(y, X)
    titanic_model = train_model(train_X, train_y)
    titanic_preds(titanic_model, val_X, val_y)

    #Predictions
    data_test = encode_sex(DATA_TEST)
    X_test = prepare_data_test(data_test)
    titanic_model_full = train_full_model(y,X)

    predict_test(titanic_model_full, X_test, data_test)
