# NASA Volumetric Soil Moisture Model (NASA Space Apps Challenge 2024)

For this year's NASA Space Apps Challenge, our challenge was to leverage earth observation data for informed agricultural decision-making. The dataset was sourced from NASA's Giovanni tool where we collected 3 year's worth of monthly averaged volumetric soil moisture data in North America. That equates to over 3 million data points.

Tools/Technologies Used:

Python
scikit-learn
joblib
Google Cloud Platform
Google Cloud Storage
Google Cloud Run
Flask

The Model

We leveraged a random forest regressor hoping that it would pick up spatial patterns in the data. The model was trained on half of the dataset (1.5 million data points) due to computational and time limitations (primarily CPU). Grid search and a 5-fold cross-validation was used for hyperparameter tuning. We produced an $$R^2$$ value of 0.912 which means that our model explains a large amount of the variance in the data which was promising. Our model also produced an accuracy rate of 85% given a 7% threshold. 

Website: https://soilvantage-bc523.web.app/ 
