from sklearn.ensemble import RandomForestRegressor

def train_model(X, y):
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model
