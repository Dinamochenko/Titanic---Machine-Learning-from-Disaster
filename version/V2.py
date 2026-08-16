import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin                                                                                                                                                    
from sklearn.pipeline import Pipeline

DATA_TRAIN = pd.read_csv('../dataset/train.csv')
DATA_TEST = pd.read_csv('../dataset/test.csv')

TITLE = ['Mr.', 'Miss.', 'Mrs.', 'Master.', 
         'Dr.', 'Rev.', 'Major.', 'Col.', 'Capt.',
         'Mlle.', 'Mme.', 'Ms.', 'Don.', 'Lady.', 'Sir.',
         'Countess.', 'Jonkheer.', 'Dona.'] 

class AgeImputerByTitle(BaseEstimator, TransformerMixin):                                                                                                                                                   
    def fit(self, X, y=None):                                                                                                                                                                                                                                                                                                                
        self.avg_age_ = X.groupby('Title')['Age'].mean()                                                                                                                                         
        self.fallback_ = X['Age'].mean()                                                                                                                                    
        return self                                                                                                                               

    def transform(self, X):                                                                                                                                                                                 
        X = X.copy()                                                                                                                                      

        X['Age'] =  X['Age'].fillna(X['Title'].map(self.avg_age_))
        X['Age'] = X['Age'].fillna(self.fallback_)
        X = X.drop(columns=['Title'])                                                                                                                       
        return X  

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

def is_familly(data):
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    return data

def cross_validation_model(pipe,X, y):
    param_grid = {'model__n_estimators': [10, 50, 100], 
                  'model__max_depth': [2, 5, 10],
                  'model__min_samples_leaf': [2, 5, 10]}

    grid = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=5)
    grid.fit(X, y)
    resultat = pd.DataFrame(grid.cv_results_)

    best = resultat['mean_test_score'].max()
    best_std = resultat.loc[resultat['mean_test_score'].idxmax(), 'std_test_score']
    seuil = best - best_std
    candidats = resultat[ resultat['mean_test_score'] >= seuil]
    choice = candidats.sort_values('param_model__max_depth').iloc[0]

    return choice.params

def compute_features_full(data, features_list):
    features = features_list
    X = data[features]

    return X

def compute_features_pipeline(data, features_list):
    y = data.Survived
    features = features_list
    X = data[features]

    return X, y

def pre_preprocess_pipeline(data):
    standardized_data = encode_sex(data)
    standardized_data = is_familly(standardized_data)
    standardized_data = name_to_title(standardized_data)
    X, y = compute_features_pipeline(standardized_data,['Sex', 'Pclass', 'Age', 'FamilySize', 'Title'])

    return X, y, standardized_data

def pre_preprocess_test(data):
    standardized_data = encode_sex(data)
    standardized_data = is_familly(standardized_data)
    standardized_data = name_to_title(standardized_data)
    X = compute_features_full(standardized_data,['Sex', 'Pclass', 'Age', 'FamilySize', 'Title'])

    return X, standardized_data

def run(pipe, X_familly, y_familly, X, standardized_data):
    pipe.fit(X_familly, y_familly)
    titanic_preds = pipe.predict(X)
        
    output = pd.DataFrame({'PassengerId': standardized_data.PassengerId,
                        'Survived': titanic_preds})
    output.to_csv('../submission/submission_V2.csv', index=False)

if __name__ == "__main__":
    
    X_familly, y_familly, standardized_data = pre_preprocess_pipeline(DATA_TRAIN)
    pipe = Pipeline([
        ('age', AgeImputerByTitle()),
        ('model', RandomForestClassifier(random_state=1)) 
    ])
    choice = cross_validation_model(pipe, X_familly, y_familly)
    pipe.set_params(**choice)
    X, standardized_data = pre_preprocess_test(DATA_TEST)
    run(pipe, X_familly, y_familly, X, standardized_data)