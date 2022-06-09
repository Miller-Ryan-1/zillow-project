# wrangle
from acquire import get_zillow_data
import pandas as pd
import numpy as np
from scipy import stats
import sklearn.preprocessing

import warnings
warnings.filterwarnings("ignore")

def wrangle_zillow():
    '''
    Acquires and prepares zillow data for exploration and modeling.
    Function Actions: Pulls Data -> Drops Quality Column  -> Drops Nulls -> Converts datatypes to int (where possible) -> eliminates odd values
    '''
    # Pull data using an acquire function
    df = get_zillow_data()
    df_size = df.shape[0]

    # Drop Quality Score
    df = df.drop(columns = 'quality')

    # Fill in Missing Values
    df['fireplaces'] = df.fireplaces.fillna(value=0)
    df['pools'] = df.pools.fillna(value=0)
    df['garages'] = df.garages.fillna(value=0)

    # Drop all nulls from dataset
    df = df.dropna()

    # Convert to integers where we can
    df = df.astype({'bedrooms':'int', 'bathrooms':'int','sqft':'int', 'selling_price':'int', 'yearbuilt':'int','fireplaces':'int','lotsize':'int','pools':'int','garages':'int'})

    # Eliminate the funky values
    df = df[df['bathrooms'] > 0]
    df = df[df['bedrooms'] > 0]
    df = df[df['bathrooms'] < 7]
    df = df[df['bedrooms'] < 7]
    df = df[df['sqft'] > 400]
    df = df[df['sqft'] < 10000]
    df = df[df['selling_price'] > 10000]
    df = df[df['selling_price'] < 10000000]
    df = df[df['fireplaces'] < 4]
    df = df[df['garages'] < 5]
    df = df[df['lotsize'] < 1000000]
    df = df[df['lotsize'] > (.5 * df['sqft'])]
    print(f'{(100*df.shape[0]/df_size):.2f}% of records remain after cleaning.')
    print(f'\n-----\n')

    # Convert Fips to Names
    df['fips_name'] = np.where(df.fips == 6037, 'Los Angeles', np.where(df.fips == 6059, 'Orange','Ventura') )
    df = df.drop(columns = 'fips')

    return df

def wrangle_zillow_quality():
    '''
    Acquires and prepares zillow data for exploration and modeling LEAVING QUALITY as a feature.
    Function Actions: Pulls Data -> Drops Nulls -> Converts datatypes to int (where possible) -> eliminates odd values
    '''
    # Pull data using an acquire function
    df = get_zillow_data()

    # Drop all nulls from dataset
    df = df.dropna()

    # Convert to integers where we can
    df = df.astype({'bedrooms':'int', 'sqft':'int', 'selling_price':'int', 'yearbuilt':'int','fireplaces':'int','lotsize':'int','pools':'int','garages':'int','fips':'int','quality':'int'})

    # Eliminate the funky values
    df = df[df['bathrooms'] > 0]
    df = df[df['bedrooms'] > 0]
    df = df[df['bathrooms'] < 7]
    df = df[df['bedrooms'] < 7]
    df = df[df['sqft'] > 400]
    df = df[df['sqft'] < 10000]
    df = df[df['selling_price'] > 10000]
    df = df[df['selling_price'] < 10000000]
    df = df[df['fireplaces'] < 4]
    df = df[df['garages'] < 5]
    df = df[df['lotsize'] < 1000000]
    df = df[df['lotsize'] > (.5 * df['sqft'])]

    # Convert Fips to Names
    df['fips_name'] = np.where(df.fips == 6037, 'Los Angeles', np.where(df.fips == 6059, 'Orange','Ventura') )
    df = df.drop(columns = 'fips')

    return df

def scale_zillow(df_train,df_validate,df_test):
    # Create the object
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(df_train.drop(columns=['fips_name','selling_price']))

    # Fit the data
    df_train_scaled = pd.DataFrame(scaler.transform(df_train.drop(columns=['fips_name','selling_price'])),columns=df_train.drop(columns=['fips_name','selling_price']).columns.values).set_index([df_train.index.values])
    df_validate_scaled = pd.DataFrame(scaler.transform(df_validate.drop(columns=['fips_name','selling_price'])),columns=df_validate.drop(columns=['fips_name','selling_price']).columns.values).set_index([df_validate.index.values])
    df_test_scaled = pd.DataFrame(scaler.transform(df_test.drop(columns=['fips_name','selling_price'])),columns=df_test.drop(columns=['fips_name','selling_price']).columns.values).set_index([df_test.index.values])

    # Add back in the fips
    df_train_scaled['fips_name'] = df_train['fips_name']
    df_validate_scaled['fips_name'] = df_validate['fips_name']
    df_test_scaled['fips_name'] = df_test['fips_name']

    # Add back in the target, selling_price
    df_train_scaled['selling_price'] = df_train['selling_price']
    df_validate_scaled['selling_price'] = df_validate['selling_price']
    df_test_scaled['selling_price'] = df_test['selling_price']

    # Encode fips_name
    dummy_df_train = pd.get_dummies(df_train_scaled[['fips_name']], dummy_na=False, drop_first=False)
    dummy_df_validate = pd.get_dummies(df_validate_scaled[['fips_name']], dummy_na=False, drop_first=False)
    dummy_df_test = pd.get_dummies(df_test_scaled[['fips_name']], dummy_na=False, drop_first=False)
    
    df_train_scaled = pd.concat([df_train_scaled, dummy_df_train], axis=1)
    df_validate_scaled = pd.concat([df_validate_scaled, dummy_df_validate], axis=1)
    df_test_scaled = pd.concat([df_test_scaled, dummy_df_test], axis=1)

    return df_train_scaled, df_validate_scaled, df_test_scaled

