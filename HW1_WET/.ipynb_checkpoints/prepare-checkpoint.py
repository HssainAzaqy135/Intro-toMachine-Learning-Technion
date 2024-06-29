import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def prepare_data(training_data, new_data):
    new_copy = new_data.copy()
    # removing blood_type and adding special_property
    new_copy["SpecialProperty"] = new_copy["blood_type"].isin(["O+", "B+"])
    new_copy.drop(columns=["blood_type"], inplace=True)
    
    # Splitting PCR features
    all_features = new_data.columns.tolist() 
    pcr_features = [feature for feature in all_features if feature.startswith("PCR")]
    list_to_standard_scale = [feature for feature in pcr_features if (pcr_features.index(feature)+1) in [1,2,4,5,6,7,8,9]]
    list_to_min_max = [feature for feature in pcr_features if (pcr_features.index(feature)+1) in [3,10]]

    # features to scale
    features_to_min_max = new_data[list_to_min_max]
    features_to_standard_scale = new_data[list_to_standard_scale]
    
    # features to fit the scalers
    min_max_fit_data = training_data[list_to_min_max]
    standard_scale_fit_data = training_data[list_to_standard_scale]
    
    # init scalers
    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    standard_scaler = StandardScaler()
    
    # Fitting scalers to training data
    min_max_scaler.fit(min_max_fit_data)
    standard_scaler.fit(standard_scale_fit_data)
    
    # scaling
    min_max_scaled_features = pd.DataFrame(min_max_scaler.transform(features_to_min_max),
                                           columns=list_to_min_max,
                                           index=new_copy.index)
    standard_scaled_features = pd.DataFrame(standard_scaler.transform(features_to_standard_scale),
                                            columns=list_to_standard_scale,
                                            index=new_copy.index)
    
    # updating the needed dataframe
    new_copy.update(min_max_scaled_features)
    new_copy.update(standard_scaled_features)


    return new_copy


    