import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

def evaluate():
    test = pd.read_csv('test.csv')
    X_test = test[['feature1', 'feature2']]
    y_test = test['label']
    
    model = joblib.load('model.pkl')
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {acc:.2f}')

if __name__ == '__main__':
    evaluate()