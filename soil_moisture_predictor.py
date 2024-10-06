from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import netCDF4
import joblib
import json

# Data Preparation

def clean_data():

    jan_data = [netCDF4.Dataset("jan_2022.nc"), netCDF4.Dataset("jan_2023.nc"), netCDF4.Dataset("jan_2024.nc")]
    feb_data = [netCDF4.Dataset("feb_2022.nc"), netCDF4.Dataset("feb_2023.nc"), netCDF4.Dataset("feb_2024.nc")]
    mar_data = [netCDF4.Dataset("mar_2022.nc"), netCDF4.Dataset("mar_2023.nc"), netCDF4.Dataset("mar_2024.nc")]
    apr_data = [netCDF4.Dataset("apr_2022.nc"), netCDF4.Dataset("apr_2023.nc"), netCDF4.Dataset("apr_2024.nc")]
    may_data = [netCDF4.Dataset("may_2022.nc"), netCDF4.Dataset("may_2023.nc"), netCDF4.Dataset("may_2024.nc")]
    jun_data = [netCDF4.Dataset("jun_2022.nc"), netCDF4.Dataset("jun_2023.nc"), netCDF4.Dataset("jun_2024.nc")]
    jul_data = [netCDF4.Dataset("jul_2022.nc"), netCDF4.Dataset("jul_2023.nc"), netCDF4.Dataset("jul_2024.nc")]
    aug_data = [netCDF4.Dataset("aug_2022.nc"), netCDF4.Dataset("aug_2023.nc"), netCDF4.Dataset("aug_2024.nc")]
    sep_data = [netCDF4.Dataset("sep_2022.nc"), netCDF4.Dataset("sep_2023.nc"), netCDF4.Dataset("sep_2024.nc")]
    oct_data = [netCDF4.Dataset("oct_2021.nc"), netCDF4.Dataset("oct_2022.nc"), netCDF4.Dataset("oct_2023.nc")]
    nov_data = [netCDF4.Dataset("nov_2021.nc"), netCDF4.Dataset("nov_2022.nc"), netCDF4.Dataset("nov_2023.nc")]
    dec_data = [netCDF4.Dataset("dec_2021.nc"), netCDF4.Dataset("dec_2022.nc"), netCDF4.Dataset("dec_2023.nc")]

    monthly_data = [jan_data, feb_data, mar_data, apr_data, may_data, jun_data, jul_data, aug_data, sep_data, oct_data, nov_data, dec_data]
    X, Y = [], []

    for month_data in monthly_data:
        print(monthly_data.index(month_data))
        for year in month_data:
            # North America Bounding Box
            for i in range(410, 640):
                for j in range(46, 510):
                    if year.variables['LPRM_AMSR2_D_SOILM3_001_soil_moisture_c1'][i][j] != "--":
                        X.append([i, j, monthly_data.index(month_data)])
                        Y.append(year.variables['LPRM_AMSR2_D_SOILM3_001_soil_moisture_c1'][i][j])

    cache = open('data.txt', 'w')
    cache.write(str(X) + '\n')
    cache.write(str(Y))                    
    cache.close()

# Hyperparameter Tuning

def model_tune_decision_tree():

    # Fetch data

    cache = open("data.txt", 'r')
    X = json.loads(cache.readline())
    y = json.loads(cache.readline())

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 175, shuffle = True, test_size = 0.50)

    param_grid = {
    'n_estimators': [100, 200],  
    'max_depth': [10, 20],     
    'min_samples_split': [5, 10],       
    'min_samples_leaf': [5, 10],
    'max_features': ['sqrt', 'log2']  
    }

    rf = RandomForestRegressor(random_state=42)

    grid_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, 
                           cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)

    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2_score = best_rf.score(X_test, y_test)

    print(f"Mean Squared Error: {mse}")
    print(f"R² Score: {r2_score}")

# Saving Model

def save_model():

    cache = open("data.txt", 'r')
    X = json.loads(cache.readline())
    y = json.loads(cache.readline())

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 175, shuffle = True, test_size = 0.25)

    rf = RandomForestRegressor(n_estimators = 225, max_depth = 20, min_samples_split = 5, min_samples_leaf = 5, max_features = 'log2', random_state=20)
    rf.fit(X_train, y_train)
    joblib.dump(rf, 'random_forest_model.pkl', compress=9)

# Model Predicting

def model_predict(lat, long, month):

    loaded_rf = joblib.load('random_forest_model.pkl')
    y_pred = loaded_rf.predict([[lat, long, month]])

    return round(y_pred[0])

save_model()