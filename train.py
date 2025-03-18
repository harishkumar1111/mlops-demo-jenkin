import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

def train():
    train = pd.read_csv('train.csv')
    X = train[['feature1', 'feature2']]
    y = train['label']
    
    model = RandomForestClassifier()
    model.fit(X, y)
    joblib.dump(model, 'model.pkl')
    print("Model trained and saved.")

if __name__ == '__main__':
    train()