# Volumetric Soil Moisture Model (2024 NASA Space Apps Challenge)

Completed with Aistis Meiklejohn, Abtin Turing, and Reid Playter.

### Overview

Our challenge for this year's NASA Space Apps Challenge was to leverage earth observation data for informed agricultural decision-making. The dataset was sourced from NASA's Giovanni tool where we collected 3 year's worth of monthly averaged volumetric soil moisture data in North America. That equates to over 3 million data points.

### Tools and Technologies

- Python
- scikit-learn
- joblib
- Google Cloud Platform
- Google Cloud Storage
- Google Cloud Run
- Flask

### The Model

We leveraged a random forest regressor hoping it would pick up spatial patterns in the data. The model was trained on half of the dataset (1.5 million data points) due to computational and time limitations (primarily CPU). Grid search and a 5-fold cross-validation were used for hyperparameter tuning. We produced an $$R^2$$ value of 0.912 which means that our model explains a large amount of the variance in the data which was promising. Our model also produced an accuracy rate of 85% given a 7% threshold. 

We hosted our model on Google Cloud Platform using Cloud Storage and Cloud Run. With Flask, we then built a REST API for soil moisture estimates to be generated from our front-end.  

Website: https://soilvantage-bc523.web.app/ 

![alt text](https://github.com/jobcabanto/NASA-Soil-Moisture-Model/blob/main/Screenshot%202024-10-15%20161901.png) 
