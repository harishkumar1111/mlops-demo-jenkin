import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess():
    # Generate sample data
    data = pd.DataFrame({'feature1': range(100), 'feature2': range(100, 200), 'label': [1 if x % 2 == 0 else 0 for x in range(100)]})
    
    # Split data
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    train.to_csv('train.csv', index=False)
    test.to_csv('test.csv', index=False)

if __name__ == '__main__':
    preprocess()