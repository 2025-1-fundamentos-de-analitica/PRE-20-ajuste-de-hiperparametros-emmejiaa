#
# Empaquetado del entrenamiento del modelo
#
def train_estimator(alpha=0.5, l1_ratio=0.5, verbose=1):

    import os
    import pickle

    import pandas as pd
    from sklearn.linear_model import ElasticNet
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=";")

    y = df["quality"]
    x = df.copy()
    x.pop("quality")

    (x_train, x_test, y_train, y_test) = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=0,
    )

    estimator = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=12345)

    estimator.fit(x_train, y_train)
    y_pred = estimator.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    if verbose > 0:
        print(estimator, ":", sep="")
        print(f"  MSE: {mse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")

    if not os.path.exists("estimator.pickle"):
        saved_estimator = None
    else:
        with open("estimator.pickle", "rb") as file:
            saved_estimator = pickle.load(file)

    if saved_estimator is None or estimator.score(
        x_test, y_test
    ) > saved_estimator.score(x_test, y_test):
        with open("estimator.pickle", "wb") as file:
            pickle.dump(estimator, file)



######
#
# Experimiento
#
train_estimator(0.2, 0.2)





#
# Experimiento
#
train_estimator(0.5, 0.5)



#####

#
# Uso del modelo en productivo
#
def use_estimator():

    import pandas as pd
    import pickle

    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=";")

    y = df["quality"]
    x = df.copy()
    x.pop("quality")

    with open("estimator.pickle", "rb") as file:
        estimator = pickle.load(file)

    y_pred = estimator.predict(x)

    return y_pred

use_estimator()












########

#
# Carga de datos
#
def load_data():

    import pandas as pd

    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=";")

    y = df["quality"]
    x = df.copy()
    x.pop("quality")

    return x, y
    
    
    
#
# Particionamiento de datos
#
def make_train_test_split(x, y):

    from sklearn.model_selection import train_test_split

    (x_train, x_test, y_train, y_test) = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=0,
    )
    return x_train, x_test, y_train, y_test    
    
    
    
#
# Cálculo de metricas de evaluación
#
def eval_metrics(y_true, y_pred):

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return mse, mae, r2
    
    
    
    
    
    
#
# Reporte de métricas de evaluación
#
def report(estimator, mse, mae, r2):

    print(estimator, ":", sep="")
    print(f"  MSE: {mse}")
    print(f"  MAE: {mae}")
    print(f"  R2: {r2}")
    
    
    
    
    
    
#
# Almacenamiento del modelo
#
def save_best_estimator(estimator):

    import os
    import pickle

    with open("estimator.pickle", "wb") as file:
        pickle.dump(estimator, file)
        
        
        
        
        
        

#
# Carga del modelo
#
def load_best_estimator():

    import os
    import pickle

    if not os.path.exists("estimator.pickle"):
        return None
    with open("estimator.pickle", "rb") as file:
        estimator = pickle.load(file)

    return estimator
    
    
    
    
    
    
#
# Entrenamiento
#
def train_estimator(alpha=0.5, l1_ratio=0.5, verbose=1):

    from sklearn.linear_model import ElasticNet

    x, y = load_data()
    x_train, x_test, y_train, y_test = make_train_test_split(x, y)
    estimator = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=12345)
    estimator.fit(x_train, y_train)
    mse, mae, r2 = eval_metrics(y_test, y_pred=estimator.predict(x_test))
    if verbose > 0:
        report(estimator, mse, mae, r2)

    best_estimator = load_best_estimator()
    if best_estimator is None or estimator.score(x_test, y_test) > best_estimator.score(
        x_test, y_test
    ):
        best_estimator = estimator

    save_best_estimator(best_estimator)
    
    
    
    
    
    
###########
train_estimator(0.5, 0.5)
train_estimator(0.2, 0.2)
train_estimator(0.1, 0.1)



######
def check_estimator():

    x, y = load_data()
    x_train, x_test, y_train, y_test = make_train_test_split(x, y)
    estimator = load_best_estimator()
    mse, mae, r2 = eval_metrics(y_test, y_pred=estimator.predict(x_test))
    report(estimator, mse, mae, r2)


#
# Debe coincidir con el mejor modelo encontrado en la celdas anteriores
#
check_estimator()







def make_hyperparameters_search(alphas, l1_ratios):

    for alpha in alphas:
        for l1_ratio in l1_ratios:
            train_estimator(alpha=alpha, l1_ratio=l1_ratio, verbose=0)
            
            
            
            
            

    
    
    
    
    
import numpy as np

alphas = np.linspace(0.0001, 0.5, 10)
l1_ratios = np.linspace(0.0001, 0.5, 10)
make_hyperparameters_search(alphas, l1_ratios)
check_estimator()
    
    








def train_estimator(alphas, l1_ratios, n_splits=5, verbose=1):

    from sklearn.linear_model import ElasticNet
    from sklearn.model_selection import GridSearchCV

    x, y = load_data()
    x_train, x_test, y_train, y_test = make_train_test_split(x, y)

    # -------------------------------------------------------------------------
    # Búsqueda de parámetros con validación cruzada
    #
    estimator = GridSearchCV(
        estimator=ElasticNet(
            random_state=12345,
        ),
        param_grid={
            "alpha": alphas,
            "l1_ratio": l1_ratios,
        },
        cv=n_splits,
        refit=True,
        verbose=0,
        return_train_score=False,
    )
    # -------------------------------------------------------------------------

    estimator.fit(x_train, y_train)

    estimator = estimator.best_estimator_

    mse, mae, r2 = eval_metrics(y_test, y_pred=estimator.predict(x_test))
    if verbose > 0:
        report(estimator, mse, mae, r2)

    best_estimator = load_best_estimator()
    if best_estimator is None or estimator.score(x_test, y_test) > best_estimator.score(
        x_test, y_test
    ):
        best_estimator = estimator

    save_best_estimator(best_estimator)
    
    
    
    
    
    

###########
import numpy as np

train_estimator(
    alphas=np.linspace(0.0001, 0.5, 10),
    l1_ratios=np.linspace(0.0001, 0.5, 10),
    n_splits=5,
    verbose=1,
)



check_estimator()





