import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn import metrics as me
from sklearn.preprocessing import StandardScaler

def metricas(texto,y_teste,y_pred):
    mae = me.mean_absolute_error(y_teste, y_pred)
    mse = me.mean_squared_error(y_teste, y_pred)
    rmse = np.sqrt(mse)
    r2 = me.r2_score(y_teste, y_pred)
    print("\n------------------------------------",texto)
    print("------------------------------------")
    print(f"MAE ...............: {mae:.4f}")
    print(f"MSE ...............: {mse:.4f}")
    print(f"RMSE ..............: {rmse:.4f}")
    print(f"R² ................: {r2:.4f}")
    return {"metodo": texto, "mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


# ==============================================================
# LEITURA DO ARQUIVO
# ==============================================================
def ler_arquivo(arquivo):
    """Lê um .txt (CSV sem cabeçalho). Última coluna = valor a prever."""
    df = pd.read_csv(arquivo, header=None, skipinitialspace=True).dropna()
    if df.shape[1] < 2:
        raise ValueError("O arquivo precisa ter ao menos 1 atributo e 1 alvo.")
    try:
        df = df.astype(float)
    except ValueError:
        raise ValueError("Na regressão todas as colunas, inclusive a última, devem ser numéricas.")
    atributos = df.iloc[:, :-1].values
    target = df.iloc[:, -1].values
    return atributos, target

def mlp_treino(x_train, y_train):
    mlp_model = MLPRegressor(
        hidden_layer_sizes=(150,),
        activation='relu',
        solver='lbfgs',
        alpha=0.01,
        learning_rate='adaptive',
        max_iter=1000,
        random_state=42
    )
    mlp_model.fit(x_train, y_train)
    return mlp_model

def mlp_teste(x_teste, y_teste, mlp_model):
    y_pred = mlp_model.predict(x_teste)
    return metricas("MLP",y_teste,y_pred)
    
def random_forest_treino(x_train, y_train):
    rf_model = RandomForestRegressor(
        n_estimators=80,
        criterion='squared_error',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(x_train, y_train)
    return rf_model

def random_forest_teste(x_teste, y_teste, rf_model):
    y_pred = rf_model.predict(x_teste)
    return metricas("RANDOM FOREST",y_teste,y_pred)

def knn_treino(x_train, y_train):
    knn_model = KNeighborsRegressor(
        n_neighbors=5,
        weights='uniform',
        metric='minkowski',
        n_jobs=-1
    )
    knn_model.fit(x_train, y_train)
    return knn_model

def knn_teste(x_teste, y_teste, knn_model):
    y_pred = knn_model.predict(x_teste)
    return metricas("KNN",y_teste,y_pred)

def svr_treino(x_train, y_train):
    svr_model = SVR(
        kernel='rbf',
        C=1.0,
        epsilon=0.1,
        gamma='scale'
    )
    svr_model.fit(x_train, y_train)
    return svr_model

def svr_teste(x_teste, y_teste, svr_model):
    y_pred = svr_model.predict(x_teste)
    return metricas("SVR",y_teste,y_pred)

def lr_treino(x_train, y_train):
    lr_model = LinearRegression()
    lr_model.fit(x_train, y_train)
    return lr_model

def lr_teste(x_teste, y_teste, lr_model):
    y_pred = lr_model.predict(x_teste)
    return metricas("REGRESSÃO LINEAR",y_teste,y_pred)

def ada_treino(x_train, y_train):
    ada_model = AdaBoostRegressor(
        n_estimators=100,
        learning_rate=1.0,
        random_state=42
    )
    ada_model.fit(x_train, y_train)
    return ada_model

def ada_teste(x_teste, y_teste, ada_model):
    y_pred = ada_model.predict(x_teste)
    return metricas("ADABOOST",y_teste,y_pred)

# ==============================================================
# INTEGRAÇÃO COM A INTERFACE (interface.py)
# ==============================================================
# nome exibido : (função de treino, função de teste, usa normalização?)
METODOS = {
    "MLP": (mlp_treino, mlp_teste, True),
    "Random Forest": (random_forest_treino, random_forest_teste, False),
    "KNN": (knn_treino, knn_teste, True),
    "SVR": (svr_treino, svr_teste, True),
    "Regressão Linear": (lr_treino, lr_teste, False),
    "AdaBoost": (ada_treino, ada_teste, False),
}

def preparar(feat, target, pct_teste):
    x_treino, x_teste, y_treino, y_teste = train_test_split(
        feat, target, test_size=pct_teste/100, random_state=42
    )
    scaler = StandardScaler()
    x_treino_sc = scaler.fit_transform(x_treino)
    x_teste_sc = scaler.transform(x_teste)
    return {"x_treino": x_treino, "x_teste": x_teste,
            "x_treino_sc": x_treino_sc, "x_teste_sc": x_teste_sc,
            "y_treino": y_treino, "y_teste": y_teste}

def executar(nome, dados):
    treino, teste, normaliza = METODOS[nome]
    if normaliza:
        x_tr, x_te = dados["x_treino_sc"], dados["x_teste_sc"]
    else:
        x_tr, x_te = dados["x_treino"], dados["x_teste"]
    modelo = treino(x_tr, dados["y_treino"])
    resultado = teste(x_te, dados["y_teste"], modelo)
    resultado["metodo"] = nome
    return resultado

# ==============================================================
# MÓDULO PRINCIPAL
# ==============================================================
if __name__ == "__main__":
    arquivo = "diabetes.txt"

    feat, target = ler_arquivo(arquivo)

    x_treino, x_teste, y_treino, y_teste = train_test_split(
        feat,target,test_size=0.2,random_state=42
    )

    scaler = StandardScaler()
    x_treino_sc = scaler.fit_transform(x_treino)
    x_teste_sc = scaler.transform(x_teste)

    modelo = mlp_treino(x_treino_sc,y_treino)
    mlp_teste(x_teste_sc,y_teste,modelo)

    modelo = random_forest_treino(x_treino,y_treino)
    random_forest_teste(x_teste,y_teste,modelo)




