import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from splitter import splitter

def zillow_model_comparator(X_train,y_train,X_validate,y_validate):
    '''
    This function creates a DataFrame to compare a number of linear regression models.
    These were done piecemeal in the zillow workbook, and were then cut and pasted in here.
    This is specific to the zillow report.  Note that it takes in the training and validating data.
    '''
    model_comparator = []
    # We need y_train and y_validate to be dataframes to append the new columns with predicted values. 
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)

    # 1. Predict target_pred_mean
    selling_price_pred_mean = y_train['selling_price'].mean()
    y_train['selling_price_pred_mean'] = selling_price_pred_mean
    y_validate['selling_price_pred_mean'] = selling_price_pred_mean

    # 2. compute target_pred_median
    selling_price_pred_median = y_train['selling_price'].median()
    y_train['selling_price_pred_median'] = selling_price_pred_median
    y_validate['selling_price_pred_median'] = selling_price_pred_median

    # 3. RMSE of target_pred_mean
    rmse_train_mean = mean_squared_error(y_train.selling_price, y_train.selling_price_pred_mean)**(1/2)
    rmse_validate_mean = mean_squared_error(y_validate.selling_price, y_validate.selling_price_pred_mean)**(1/2)

    # 4. RMSE of target_pred_median
    rmse_train_median = mean_squared_error(y_train.selling_price, y_train.selling_price_pred_median)**(1/2)
    rmse_validate_median = mean_squared_error(y_validate.selling_price, y_validate.selling_price_pred_median)**(1/2)

    results = {'Model' : 'Baseline Median','RMSE Train':rmse_train_median,'RMSE Validate':rmse_validate_median}
    model_comparator.append(results)
    results = {'Model' : 'Baseline Mean','RMSE Train':rmse_train_mean,'RMSE Validate':rmse_validate_mean}
    model_comparator.append(results)

    # create the model object
    lm = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm.fit(X_train, y_train.selling_price)

    # predict train
    y_train['selling_price_pred_lm'] = lm.predict(X_train)

    # evaluate: rmse
    rmse_train_ols = mean_squared_error(y_train.selling_price, y_train.selling_price_pred_lm)**(1/2)

    # predict validate
    y_validate['selling_price_pred_lm'] = lm.predict(X_validate)

    # evaluate: rmse
    rmse_validate_ols = mean_squared_error(y_validate.selling_price, y_validate.selling_price_pred_lm)**(1/2)

    results = {'Model' : 'OLS','RMSE Train':rmse_train_ols,'RMSE Validate':rmse_validate_ols}
    model_comparator.append(results)

    # create the model object
    lars = LassoLars(alpha=2.0)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lars.fit(X_train, y_train.selling_price)

    # predict train
    y_train['selling_price_pred_lars'] = lars.predict(X_train)

    # evaluate: rmse
    rmse_train_lassolars = mean_squared_error(y_train.selling_price, y_train.selling_price_pred_lars)**(1/2)

    # predict validate
    y_validate['selling_price_pred_lars'] = lars.predict(X_validate)

    # evaluate: rmse
    rmse_validate_lassolars = mean_squared_error(y_validate.selling_price, y_validate.selling_price_pred_lars)**(1/2)

    results = {'Model' : 'LassoLars','RMSE Train':rmse_train_lassolars,'RMSE Validate':rmse_validate_lassolars}
    model_comparator.append(results)

    # create the model object
    glm = TweedieRegressor(power=1, alpha=0)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    glm.fit(X_train, y_train.selling_price)

    # predict train
    y_train['selling_price_pred_glm'] = glm.predict(X_train)

    # evaluate: rmse
    rmse_train_glm = mean_squared_error(y_train.selling_price, y_train.selling_price_pred_glm)**(1/2)

    # predict validate
    y_validate['selling_price_pred_glm'] = glm.predict(X_validate)

    # evaluate: rmse
    rmse_validate_glm = mean_squared_error(y_validate.selling_price, y_validate.selling_price_pred_glm)**(1/2)

    results = {'Model' : 'GLM','RMSE Train':rmse_train_glm,'RMSE Validate':rmse_validate_glm}
    model_comparator.append(results)

    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2)

    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train)

    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2 = pf.transform(X_validate)

    # create the model object
    lm2 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(X_train_degree2, y_train.selling_price)

    # predict train
    y_train['selling_price_pred_lm2'] = lm2.predict(X_train_degree2)

    # evaluate: rmse
    rmse_train_p2 = mean_squared_error(y_train.selling_price, y_train.selling_price_pred_lm2)**(1/2)

    # predict validate
    y_validate['selling_price_pred_lm2'] = lm2.predict(X_validate_degree2)

    # evaluate: rmse
    rmse_validate_p2 = mean_squared_error(y_validate.selling_price, y_validate.selling_price_pred_lm2)**(1/2)

    results = {'Model' : 'Polynomial, Deg. 2','RMSE Train':rmse_train_p2,'RMSE Validate':rmse_validate_p2}
    model_comparator.append(results)

    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=3)

    # fit and transform X_train_scaled
    X_train_degree3 = pf.fit_transform(X_train)

    # transform X_validate_scaled & X_test_scaled
    X_validate_degree3 = pf.transform(X_validate)

    # create the model object
    lm3 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm3.fit(X_train_degree3, y_train.selling_price)

    # predict train
    y_train['selling_price_pred_lm3'] = lm3.predict(X_train_degree3)

    # evaluate: rmse
    rmse_train_p3 = mean_squared_error(y_train.selling_price, y_train.selling_price_pred_lm3)**(1/2)

    # predict validate
    y_validate['selling_price_pred_lm3'] = lm3.predict(X_validate_degree3)

    # evaluate: rmse
    rmse_validate_p3 = mean_squared_error(y_validate.selling_price, y_validate.selling_price_pred_lm3)**(1/2)

    results = {'Model' : 'Polynomial, Deg. 3','RMSE Train':rmse_train_p3,'RMSE Validate':rmse_validate_p3}
    model_comparator.append(results)

    df_results = pd.DataFrame(model_comparator).round(2).set_index('Model')
    df_results['Overfit %'] = (100*(df_results['RMSE Validate'] - df_results['RMSE Train'])/df_results['RMSE Train']).round(2)  
    return df_results.T


def count_filtered(df):
    '''
    This function creates a DataFrames for eah county comparing a number of linear regression models.
    This was done in the zillow workbook, and was then cut and pasted in here.
    This is specific to the zillow report.  Note that it takes in the wrangled dataframe with extraneous features already dropped.
    '''

    df_o = df[df['fips_name'] == 'Orange']
    df_v = df[df['fips_name'] == 'Ventura']
    df_la = df[df['fips_name'] == 'Los Angeles']
    df_o = df_o.drop(columns = 'fips_name')
    df_v = df_v.drop(columns = 'fips_name')
    df_la = df_la.drop(columns = 'fips_name')
    fips_list = [df_o,df_v,df_la]

    counter = 1

    for i in fips_list:
        model_comparator_fips = []
        # Split
        train, validate, test = splitter(i)
        # Scale
        # Create the object
        scaler = MinMaxScaler()
        scaler.fit(train.drop(columns='selling_price'))

        # Fit the data
        train_scaled = pd.DataFrame(scaler.transform(train.drop(columns=['selling_price'])),columns=train.drop(columns=['selling_price']).columns.values).set_index([train.index.values])
        validate_scaled = pd.DataFrame(scaler.transform(validate.drop(columns=['selling_price'])),columns=validate.drop(columns=['selling_price']).columns.values).set_index([validate.index.values])
        test_scaled = pd.DataFrame(scaler.transform(test.drop(columns=['selling_price'])),columns=test.drop(columns=['selling_price']).columns.values).set_index([test.index.values])

        # Add back in the target, selling_price
        train_scaled['selling_price'] = train['selling_price']
        validate_scaled['selling_price'] = validate['selling_price']
        test_scaled['selling_price'] = test['selling_price']
        # Drop fips data and seperate into target and features
        X_train = train_scaled.drop(columns='selling_price')
        y_train = train_scaled.selling_price

        X_validate = validate_scaled.drop(columns='selling_price')
        y_validate = validate_scaled.selling_price

        X_test = test_scaled.drop(columns='selling_price')
        y_test = test_scaled.selling_price
        # Run Tests!
        # We need y_train and y_validate to be dataframes to append the new columns with predicted values. 
        y_train = pd.DataFrame(y_train)
        y_validate = pd.DataFrame(y_validate)

        # 1. Predict target_pred_mean
        selling_price_pred_mean = y_train['selling_price'].mean()
        y_train['selling_price_pred_mean'] = selling_price_pred_mean
        y_validate['selling_price_pred_mean'] = selling_price_pred_mean

        # 2. compute target_pred_median
        selling_price_pred_median = y_train['selling_price'].median()
        y_train['selling_price_pred_median'] = selling_price_pred_median
        y_validate['selling_price_pred_median'] = selling_price_pred_median

        # 3. RMSE of target_pred_mean
        rmse_train_mean = mean_squared_error(y_train.selling_price, y_train.selling_price_pred_mean)**(1/2)
        rmse_validate_mean = mean_squared_error(y_validate.selling_price, y_validate.selling_price_pred_mean)**(1/2)


        # 4. RMSE of target_pred_median
        rmse_train_median = mean_squared_error(y_train.selling_price, y_train.selling_price_pred_median)**(1/2)
        rmse_validate_median = mean_squared_error(y_validate.selling_price, y_validate.selling_price_pred_median)**(1/2)

        results = {'Model' : 'Baseline Median','RMSE Train':rmse_train_median,'RMSE Validate':rmse_validate_median}
        model_comparator_fips.append(results)
        results = {'Model' : 'Baseline Mean','RMSE Train':rmse_train_mean,'RMSE Validate':rmse_validate_mean}
        model_comparator_fips.append(results)
        # create the model object
        lm = LinearRegression(normalize=True)

        # fit the model to our training data. We must specify the column in y_train, 
        # since we have converted it to a dataframe from a series! 
        lm.fit(X_train, y_train.selling_price)

        # predict train
        y_train['selling_price_pred_lm'] = lm.predict(X_train)

        # evaluate: rmse
        rmse_train_ols = mean_squared_error(y_train.selling_price, y_train.selling_price_pred_lm)**(1/2)

        # predict validate
        y_validate['selling_price_pred_lm'] = lm.predict(X_validate)

        # evaluate: rmse
        rmse_validate_ols = mean_squared_error(y_validate.selling_price, y_validate.selling_price_pred_lm)**(1/2)

        results = {'Model' : 'OLS','RMSE Train':rmse_train_ols,'RMSE Validate':rmse_validate_ols}
        model_comparator_fips.append(results)
        # create the model object
        lars = LassoLars(alpha=2.0)

        # fit the model to our training data. We must specify the column in y_train, 
        # since we have converted it to a dataframe from a series! 
        lars.fit(X_train, y_train.selling_price)

        # predict train
        y_train['selling_price_pred_lars'] = lars.predict(X_train)

        # evaluate: rmse
        rmse_train_lassolars = mean_squared_error(y_train.selling_price, y_train.selling_price_pred_lars)**(1/2)

        # predict validate
        y_validate['selling_price_pred_lars'] = lars.predict(X_validate)

        # evaluate: rmse
        rmse_validate_lassolars = mean_squared_error(y_validate.selling_price, y_validate.selling_price_pred_lars)**(1/2)

        results = {'Model' : 'LassoLars','RMSE Train':rmse_train_lassolars,'RMSE Validate':rmse_validate_lassolars}
        model_comparator_fips.append(results)
        # create the model object
        glm = TweedieRegressor(power=1, alpha=0)

        # fit the model to our training data. We must specify the column in y_train, 
        # since we have converted it to a dataframe from a series! 
        glm.fit(X_train, y_train.selling_price)

        # predict train
        y_train['selling_price_pred_glm'] = glm.predict(X_train)

        # evaluate: rmse
        rmse_train_glm = mean_squared_error(y_train.selling_price, y_train.selling_price_pred_glm)**(1/2)

        # predict validate
        y_validate['selling_price_pred_glm'] = glm.predict(X_validate)

        # evaluate: rmse
        rmse_validate_glm = mean_squared_error(y_validate.selling_price, y_validate.selling_price_pred_glm)**(1/2)

        results = {'Model' : 'GLM','RMSE Train':rmse_train_glm,'RMSE Validate':rmse_validate_glm}
        model_comparator_fips.append(results)
        # make the polynomial features to get a new set of features
        pf = PolynomialFeatures(degree=2)

        # fit and transform X_train_scaled
        X_train_degree2 = pf.fit_transform(X_train)

        # transform X_validate_scaled & X_test_scaled
        X_validate_degree2 = pf.transform(X_validate)
        
        # create the model object
        lm2 = LinearRegression(normalize=True)

        # fit the model to our training data. We must specify the column in y_train, 
        # since we have converted it to a dataframe from a series! 
        lm2.fit(X_train_degree2, y_train.selling_price)

        # predict train
        y_train['selling_price_pred_lm2'] = lm2.predict(X_train_degree2)

        # evaluate: rmse
        rmse_train_p2 = mean_squared_error(y_train.selling_price, y_train.selling_price_pred_lm2)**(1/2)

        # predict validate
        y_validate['selling_price_pred_lm2'] = lm2.predict(X_validate_degree2)

        # evaluate: rmse
        rmse_validate_p2 = mean_squared_error(y_validate.selling_price, y_validate.selling_price_pred_lm2)**(1/2)

        results = {'Model' : 'Polynomial, Deg. 2','RMSE Train':rmse_train_p2,'RMSE Validate':rmse_validate_p2}
        model_comparator_fips.append(results)
        # make the polynomial features to get a new set of features
        pf = PolynomialFeatures(degree=3)

        # fit and transform X_train_scaled
        X_train_degree3 = pf.fit_transform(X_train)

        # transform X_validate_scaled & X_test_scaled
        X_validate_degree3 = pf.transform(X_validate)
       
        # create the model object
        lm3 = LinearRegression(normalize=True)

        # fit the model to our training data. We must specify the column in y_train, 
        # since we have converted it to a dataframe from a series! 
        lm3.fit(X_train_degree3, y_train.selling_price)

        # predict train
        y_train['selling_price_pred_lm3'] = lm3.predict(X_train_degree3)

        # evaluate: rmse
        rmse_train_p3 = mean_squared_error(y_train.selling_price, y_train.selling_price_pred_lm3)**(1/2)

        # predict validate
        y_validate['selling_price_pred_lm3'] = lm3.predict(X_validate_degree3)

        # evaluate: rmse
        rmse_validate_p3 = mean_squared_error(y_validate.selling_price, y_validate.selling_price_pred_lm3)**(1/2)

        results = {'Model' : 'Polynomial, Deg. 3','RMSE Train':rmse_train_p3,'RMSE Validate':rmse_validate_p3}
        model_comparator_fips.append(results)
        
        if counter == 1:
            print('Orange County Results')
        elif counter == 2:
            print('Ventura County Results')
        else:
            print('Los Angeles Country Results')
        df_results2 = pd.DataFrame(model_comparator_fips).round(2).set_index('Model') 
        df_results2['Overfit %'] = (100*(df_results2['RMSE Validate'] - df_results2['RMSE Train'])/df_results2['RMSE Train']).round(2)  
        print(df_results2)
        print('\n-----\n')
        counter+=1