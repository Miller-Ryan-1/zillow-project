import pandas as pd
import os
from env import get_db_url


def get_zillow_data():
    '''
    Acquires zillow dataframe based on SQL query found below
    '''
    filename = 'zillow_data.csv'

    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col=0)
    else:
        df = pd.read_sql(
            '''
            SELECT 
                p.bedroomcnt AS bedrooms,
                p.bathroomcnt AS bathrooms,
                p.calculatedfinishedsquarefeet AS sqft,
                p.taxvaluedollarcnt AS selling_price,
                p.yearbuilt,
                p.buildingqualitytypeid AS quality,
                p.fireplacecnt AS fireplaces,
                p.lotsizesquarefeet AS lotsize,
                p.poolcnt AS pools,
                p.garagecarcnt AS garages,
                p.fips
            FROM
                properties_2017 AS p
                    RIGHT JOIN
                predictions_2017 USING (parcelid)
            WHERE
                propertylandusetypeid = 261;
            '''
            ,
            get_db_url('zillow')
        )

        df.to_csv(filename)

        return df