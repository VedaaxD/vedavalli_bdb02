#Implementing gradient boosting (regression) algorithm using sklearn
from ISLP import load_data
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error,r2_score

def load():
    boston=load_data('Boston')
    X=boston.drop(columns=['medv'])
    y=boston['medv']
    return X,y
def main():
    #load the data
    X,y=load()
    #splitting the train and test set
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
    #training the model
    gb_reg=GradientBoostingRegressor(n_estimators=50,learning_rate=0.1,max_depth=3,random_state=42)
    #fitting the model
    gb_reg.fit(X_train,y_train)
    #prediction
    y_pred=gb_reg.predict(X_test)
    mse=mean_squared_error(y_test,y_pred)
    r2=r2_score(y_test,y_pred)
    print('Gradient boosting regressor MSE: ',mse)
    print('Gradient boosting regressor R2 score: ',r2)
    #performing 10 fold cross validation for this example
    cv_mse = -cross_val_score(gb_reg, X, y, cv=10, scoring='neg_mean_squared_error')
    cv_r2 = cross_val_score(gb_reg, X, y, cv=10, scoring='r2')

    print(f'10-Fold CV mse: {cv_mse.mean():.4f} ± {cv_mse.std():.4f}')
    print(f'10-Fold CV R2 Score: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}')


if __name__=='__main__':
    main()
