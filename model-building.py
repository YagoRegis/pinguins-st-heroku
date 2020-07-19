from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import uniform, truncnorm, randint
import pandas as pd
import pickle

def target_encode(val):
	return target_mapper[val]

penguins = pd.read_csv('penguins-data.csv')

# Ordinal feature encoding
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
df = penguins.copy()
target = 'species'
encode = ['sex','island']

for col in encode:
	dummy = pd.get_dummies(df[col], prefix=col)
	df = pd.concat([df,dummy], axis=1)
	del df[col]

target_mapper = {'Adelie':0, 'Chinstrap':1, 'Gentoo':2}

df[target] = df[target].apply(target_encode)

# Separating X and Y
X = df.drop(target, axis=1)
Y = df[target]

model_params = {
    # randomly sample numbers from 4 to 204 estimators
    'n_estimators': randint(4,150),
    # normally distributed max_features, with mean .25 stddev 0.1, bounded between 0 and 1
    'max_features': truncnorm(a=0, b=1, loc=0.25, scale=0.1),
    # uniform distribution from 0.01 to 0.2 (0.01 + 0.199)
    'min_samples_split': uniform(0.01, 0.199)
}

# Build the model with random forest and search cv
rf_model = RandomForestClassifier()
clf = RandomizedSearchCV(rf_model, model_params, n_iter=100, cv=5, random_state=1)
model = clf.fit(X, Y)
best_estimator = model.best_estimator_


# Saving the model
pickle.dump(best_estimator, open('penguins_clf.pkl', 'wb'))