import pandas as pd
import numpy as np

def prepare_data(training_data, new_data):
    new_copy = new_data.copy()
    # Splitting PCR features
    all_features = training_data.columns.tolist()
    
    pcr_features = [feature for feature in all_features if feature.startswith("PCR")]
    
    features_to_standard_scale = [feature for feature in pcr_features if pcr_features.index(feature) in [5,6,8]]
    features_to_min_max = [feature for feature in pcr_features if pcr_features.index(feature) in [1,2,3,4,7,9,10]]

    pcr_training_DF  = training_data[features_to_standard_scale]
    pcr_new_DF = new_copy[features_to_min_max]
    
    # Min max Scaling
    max = 1
    min = -1
    min_maxed_features = (pcr_new_DF_ - pcr_new_DF.min(axis=0)) / (pcr_new_DF.max(axis=0) - pcr_new_DF.min(axis=0)) * (max - min) + min
    for feature in features_to_min_max:
        new_copy[feature] = min_maxed_features[feature]


    # Standard Scaling
    stds = pcr_training_DF.std()
    means = pcr_training_DF.mean()
    standardized_features = (pcr_new_DF-means)/stds
    for feature in features_to_standard_scale:
        new_copy[feature] = standardized_features[feature]

    return new_copy


    