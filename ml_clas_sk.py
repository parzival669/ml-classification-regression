import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics as me
from sklearn.preprocessing import StandardScaler

def metricas(texto,y_teste,y_pred):
    cm = me.confusion_matrix(y_teste, y_pred)
    ac = me.accuracy_score(y_teste, y_pred)
    pr = me.precision_score(y_teste, y_pred, average='macro', zero_division=0)
    re = me.recall_score(y_teste, y_pred, average='macro', zero_division=0)
    f1 = me.f1_score(y_teste, y_pred, average='macro', zero_division=0)
    print("\n---------------- ",texto)
    print("Matriz de Confusão:\n", cm)
    print(f"Acurácia...........: {ac*100:.2f}%")    
    print(f"Precisão...........: {pr*100:.2f}%")
    print(f"Recall.............: {re*100:.2f}%")
    print(f"F1-measure.........: {f1*100:.2f}%")

    # valores por classe e retorno (usados pela interface)
    pr_cl, re_cl, f1_cl, _ = me.precision_recall_fscore_support(
        y_teste, y_pred, average=None, zero_division=0)
    rotulos = np.unique(np.concatenate([np.asarray(y_teste), np.asarray(y_pred)]))
    return {"metodo": texto, "cm": cm, "rotulos": rotulos,
            "ac": ac, "pr": pr, "re": re, "f1": f1,
            "pr_cl": pr_cl, "re_cl": re_cl, "f1_cl": f1_cl}

def ler_arquivo(arquivo):
    """Lê um .txt (CSV sem cabeçalho). Última coluna = classe."""
    df = pd.read_csv(arquivo, header=None, skipinitialspace=True).dropna()
    if df.shape[1] < 2:
        raise ValueError("O arquivo precisa ter ao menos 1 atributo e 1 classe.")

    try:
        atributos = df.iloc[:, :-1].astype(float).values.tolist()
    except ValueError:
        raise ValueError("Os atributos (todas as colunas menos a última) devem ser numéricos.")
    classes = [c.strip() if isinstance(c, str) else c for c in df.iloc[:, -1].tolist()]

    return atributos, classes

def mlp_treino(x_train, y_train):  
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(125,),
        activation='logistic',
        solver='adam',
        alpha=0.01,
        learning_rate='adaptive',
        max_iter=10000
    )
    mlp_model.fit(x_train, y_train)
    
    return mlp_model

def mlp_teste(x_teste, y_teste, mlp_model): 
    y_pred = mlp_model.predict(x_teste)
    return metricas("MLP",y_teste,y_pred)

def rf_treino(x_train, y_train):
    rf_model = RandomForestClassifier(
        n_estimators=100,
        criterion='gini',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(x_train, y_train)
    return rf_model


def rf_teste(x_teste, y_teste, rf_model):

    y_pred = rf_model.predict(x_teste)
    return metricas("RANDOM FOREST",y_teste,y_pred)


def knn_treino(x_train, y_train):
    knn_model = KNeighborsClassifier(
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


def svm_treino(x_train, y_train):
    svm_model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        random_state=42
    )
    svm_model.fit(x_train, y_train)
    return svm_model


def svm_teste(x_teste, y_teste, svm_model):
    y_pred = svm_model.predict(x_teste)
    return metricas("SVM",y_teste,y_pred)


def dt_treino(x_train, y_train):
    dt_model = DecisionTreeClassifier(
        criterion='gini',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    dt_model.fit(x_train, y_train)
    return dt_model


def dt_teste(x_teste, y_teste, dt_model):
    y_pred = dt_model.predict(x_teste)
    return metricas("ÁRVORE DE DECISÃO",y_teste,y_pred)


def gb_treino(x_train, y_train):
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    gb_model.fit(x_train, y_train)
    return gb_model


def gb_teste(x_teste, y_teste, gb_model):
    y_pred = gb_model.predict(x_teste)
    return metricas("GRADIENT BOOSTING",y_teste,y_pred)


def ada_treino(x_train, y_train):
    ada_model = AdaBoostClassifier(
        n_estimators=100,
        learning_rate=1.0,
        random_state=42
    )
    ada_model.fit(x_train, y_train)
    return ada_model


def ada_teste(x_teste, y_teste, ada_model):
    y_pred = ada_model.predict(x_teste)
    return metricas("ADABOOST",y_teste,y_pred)


# **************************************************************
#            INTEGRAÇÃO COM A INTERFACE (interface.py)
# **************************************************************
# nome exibido : (função de treino, função de teste, usa normalização?)
METODOS = {
    "MLP": (mlp_treino, mlp_teste, True),
    "Random Forest": (rf_treino, rf_teste, False),
    "KNN": (knn_treino, knn_teste, True),
    "SVM": (svm_treino, svm_teste, True),
    "Árvore de Decisão": (dt_treino, dt_teste, False),
    "Gradient Boosting": (gb_treino, gb_teste, False),
    "AdaBoost": (ada_treino, ada_teste, False),
}

def preparar(feat, cl_str, pct_teste):
    classes = sorted(set(cl_str))
    cl = np.array([classes.index(c) for c in cl_str])
    try:
        x_treino, x_teste, y_treino, y_teste = train_test_split(
            feat, cl, test_size=pct_teste/100, random_state=42, stratify=cl
        )
    except ValueError:  # alguma classe com poucas amostras para estratificar
        x_treino, x_teste, y_treino, y_teste = train_test_split(
            feat, cl, test_size=pct_teste/100, random_state=42
        )
    scaler = StandardScaler()
    x_treino_sc = scaler.fit_transform(x_treino)
    x_teste_sc = scaler.transform(x_teste)
    return {"classes": classes,
            "x_treino": x_treino, "x_teste": x_teste,
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
    resultado["classes"] = [str(dados["classes"][int(i)]) for i in resultado["rotulos"]]
    return resultado


# **************************************************************
#                       MÓDULO PRINCIPAL
# **************************************************************
if __name__ == "__main__":
    arquivo = "iris.txt"  # ou winequality.txt
    feat, cl_str = ler_arquivo(arquivo)
    classes = list(set(cl_str))
    cl = np.array([classes.index(c) for c in cl_str])
    x_treino, x_teste, y_treino, y_teste = train_test_split(
        feat,
        cl,
        test_size=0.2,
        random_state=42,
        stratify=cl
    )

    # NORMALIZAÇÃO PARA MLP
    scaler = StandardScaler()
    x_treino_sc = scaler.fit_transform(x_treino)
    x_teste_sc = scaler.transform(x_teste)

    # MLP
    modelo_mlp = mlp_treino(x_treino_sc,y_treino)
    mlp_teste(x_teste_sc,y_teste,modelo_mlp)

    # RANDOM FOREST
    modelo_rf = rf_treino(x_treino,y_treino)
    rf_teste(x_teste,y_teste,modelo_rf)