import pandas as pd


DATA_TRAIN = pd.read_csv('dataset/train.csv')

def rate_survived(data):
    return round((len((data.loc[(data.Survived == 1 )]))/ len(data))*100 ,2)

def how_many_males_survived(data):

    data_analyse = len(data.loc[(data.Survived == 1 ) & (data.Sex == "male")])
    male_survived = (data_analyse / len(data.loc[data.Sex == "male"])) * 100

    return round(male_survived,2)

def how_many_females_survived(data):

    data_analyse = len(data.loc[(data.Survived == 1 ) & (data.Sex == "female")])
    female_survived = (data_analyse / len(data.loc[data.Sex == "female"])) * 100

    return round(female_survived,2)

def how_many_childs_survived(data):
    data_analyse = len(data.loc[(data.Survived == 1 ) & (data.Age <= 18)])
    childs_survived = (data_analyse / len(data.loc[(data.Age <= 18)])) * 100

    return round(childs_survived,2)

def pclass_survived(pclass, data):
    data_analyse = len(data.loc[(data.Survived == 1 ) & (data.Pclass == pclass)])
    pclass_survived = (data_analyse / len(data.loc[(data.Pclass == pclass)])) * 100

    return round(pclass_survived,2)

def age_survived(age1, age2, data):
    data_analyse = len(data.loc[(data.Survived == 1 ) & (data.Age >= age1) & (data.Age < age2)])
    age_s = (data_analyse / len(data.loc[(data.Age >= age1) & (data.Age < age2)])) * 100

    return round(age_s,2)

def sibling_survived(data):
    data_analyse = len(data.loc[(data.Survived == 1 ) & (data.SibSp > 0)])
    sibling_s = (data_analyse / len(data.loc[(data.SibSp > 0)])) * 100

    return round(sibling_s,2)

def parch_survived(data):

    data_analyse = len(data.loc[(data.Survived == 1 ) & (data.Parch > 0)])
    parch_s = (data_analyse / len(data.loc[(data.Parch > 0)])) * 100


    return round(parch_s,2)

def survived_pclass(sexe ,pclass, data):

    data_analyse = len(data.loc[(data.Survived == 1 ) & (data.Sex == sexe) & (data.Pclass == pclass)])
    sexe_survived = (data_analyse / len(data.loc[(data.Sex == sexe) & (data.Pclass == pclass)])) * 100

    return round(sexe_survived,2)

if __name__ == '__main__':

    print(f"{rate_survived(DATA_TRAIN)} %  survivors")

    print(f"{how_many_males_survived(DATA_TRAIN)} % male survivors")
    print(f"{how_many_females_survived(DATA_TRAIN)} % female survivors")
    print(f"{how_many_childs_survived(DATA_TRAIN)} % child survivors")

    print(f"{pclass_survived(1, DATA_TRAIN)} % 1 class survivors")
    print(f"{pclass_survived(2, DATA_TRAIN)} % 2 class  survivors")
    print(f"{pclass_survived(3, DATA_TRAIN)} % 3 class survivors")

    print(f"{age_survived(18, 50, DATA_TRAIN)} % 18-50 class survivors")
    print(f"{age_survived(50, (max(DATA_TRAIN.Age + 1)), DATA_TRAIN)} % 50+ class survivors")

    print(f"{sibling_survived(DATA_TRAIN)} % sibling +1 survivors")
    print(f"{parch_survived(DATA_TRAIN)} % parch +1 survivors")

    print(f"{survived_pclass('female', 1, DATA_TRAIN)} % female survivors 1st class")
    print(f"{survived_pclass('female', 2, DATA_TRAIN)} % female survivors 2nd class")
    print(f"{survived_pclass('female', 3, DATA_TRAIN)} % female survivors 3rd class")

    print(f"{survived_pclass('male', 1, DATA_TRAIN)} % male survivors 1st class")
    print(f"{survived_pclass('male', 2, DATA_TRAIN)} % male survivors 2nd class")
    print(f"{survived_pclass('male', 3, DATA_TRAIN)} % male survivors 3rd class")