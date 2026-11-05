import sys
import numpy as np

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QApplication, QWidget, QFrame, QLabel, QPushButton, QComboBox, QSpinBox,
    QLineEdit, QFileDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QAbstractItemView, QSizePolicy, QButtonGroup, QSlider, QProgressBar,
    QStackedWidget
)

import ml_clas_sk as clas
import ml_reg_sk as reg

# ==============================================================
# LEITURA DE ARQUIVO .TXT
# ==============================================================
# Formato: valores separados por vírgula, sem cabeçalho, uma instância por
# linha. A última coluna é o alvo (classe na classificação, número na regressão).
# Quem lê é o próprio módulo do modo escolhido (ml_clas_sk / ml_reg_sk).
MAX_CLASSES = 30   # acima disso o alvo provavelmente é contínuo -> regressão

def ler_txt(caminho, modulo, tipo):
    feat, alvo = modulo.ler_arquivo(caminho)
    if len(alvo) < 10:
        raise ValueError("O arquivo precisa ter pelo menos 10 instâncias.")
    if tipo == "classificacao" and len(set(alvo)) > MAX_CLASSES:
        raise ValueError("Selecione o modo REGRESSÃO")
    return feat, alvo


# ==============================================================
# EXECUÇÃO EM SEGUNDO PLANO (não trava a janela)
# ==============================================================
class Worker(QThread):
    resultado = Signal(dict)
    erro = Signal(str)

    def __init__(self, modulo, nomes, dados):
        super().__init__()
        self.modulo, self.nomes, self.dados = modulo, nomes, dados

    def run(self):
        for nome in self.nomes:
            try:
                self.resultado.emit(self.modulo.executar(nome, self.dados))
            except Exception as e:
                self.erro.emit(f"{nome}: {e}")


# ==============================================================
# ESTILO  (paleta: slate escuro + teal + âmbar)
# ==============================================================
BG_TOPO = "#0b1120"
BG_A = "#0f172a"
BG_B = "#0d3b45"      # brilho teal no canto inferior direito
PANEL = "#111b32"
CARD = "#16223d"
LINE = "#26375a"
DASH = "#344058"
TXT = "#f1f5f9"
MUTED = "#94a3b8"
DIM = "#55627a"
TEAL = "#14b8a6"
TEAL_ESC = "#0f766e"
AMBER = "#f59e0b"


def _icone_seta(cor, nome):
    """Desenha a seta (▼) do menu suspenso e devolve o caminho do PNG."""
    import tempfile, os
    from PySide6.QtGui import QPixmap, QPainter, QPolygonF
    from PySide6.QtCore import QPointF
    pm = QPixmap(24, 24); pm.fill(Qt.transparent)
    p = QPainter(pm); p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen); p.setBrush(QColor(cor))
    p.drawPolygon(QPolygonF([QPointF(6, 9), QPointF(18, 9), QPointF(12, 16)]))
    p.end()
    caminho = os.path.join(tempfile.gettempdir(), nome)
    pm.save(caminho, "PNG")
    return caminho.replace("\\", "/")


SETA = SETA_OFF = ""


def montar_estilo():
    """Precisa rodar depois de criar o QApplication (usa QPixmap)."""
    global SETA, SETA_OFF
    SETA = _icone_seta(MUTED, "txt_seta.png")
    SETA_OFF = _icone_seta(DIM, "txt_seta_off.png")
    return QSS_BASE.replace("{SETA_OFF}", SETA_OFF).replace("{SETA}", SETA)


QSS_BASE = f"""
* {{ font-family: 'Segoe UI', 'Inter', 'Helvetica Neue', Arial, sans-serif; color: {TXT}; }}
QWidget#raiz {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
        stop:0 {BG_TOPO}, stop:0.42 {BG_A}, stop:1 {BG_B});
}}
QWidget#topo {{ background: {BG_TOPO}; border-bottom: 1px solid {LINE}; }}
QLabel#logo {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 {TEAL}, stop:1 {TEAL_ESC});
    border-radius: 12px; color: white; font-size: 17px; font-weight: 800; }}
QLabel#marca {{ font-size: 16px; font-weight: 800; }}
QLabel#sub {{ font-size: 11px; color: {MUTED}; }}

QFrame#seg {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 19px; }}
QPushButton#aba {{ background: transparent; border: 1px solid transparent; border-radius: 15px;
    padding: 7px 20px; font-size: 11px; font-weight: 700; color: {MUTED}; }}
QPushButton#aba:hover {{ color: {TXT}; }}
QPushButton#aba:checked {{ background: rgba(20,184,166,0.16); border: 1px solid {TEAL}; color: {TEAL}; }}

QFrame#painel {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 18px; }}
QLabel#titulo {{ font-size: 20px; font-weight: 800; }}
QLabel#rotulo {{ font-size: 11px; font-weight: 700; color: {MUTED}; }}
QLabel#dica {{ font-size: 11px; color: {MUTED}; }}
QLabel#badge {{ background: {TEAL}; color: {BG_TOPO}; border-radius: 11px; font-size: 11px; font-weight: 800; }}

QFrame#drop {{ background: {CARD}; border: 1px dashed {DASH}; border-radius: 14px; }}
QFrame#drop:hover {{ border: 1px dashed {TEAL}; }}
QFrame#drop[arrasto="true"] {{ border: 1px solid {TEAL}; background: rgba(20,184,166,0.12); }}
QLabel#drop_icone {{ color: {TEAL}; font-size: 20px; }}
QLabel#drop_titulo {{ font-size: 13px; font-weight: 700; }}
QLabel#drop_sub {{ font-size: 11px; color: {MUTED}; }}

QFrame#stat {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 12px; }}
QLabel#stat_valor {{ font-size: 17px; font-weight: 800; }}
QLabel#stat_nome {{ font-size: 9px; font-weight: 700; color: {MUTED}; }}

QComboBox {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 12px;
    padding: 9px 14px; font-size: 13px; min-height: 20px; }}
QComboBox:focus, QComboBox:on {{ border: 1px solid {TEAL}; }}
QComboBox:disabled {{ color: {DIM}; }}
QComboBox::drop-down {{ border: none; background: transparent; width: 34px; }}
QComboBox::down-arrow {{ image: url({{SETA}}); width: 12px; height: 12px; margin-right: 8px; }}
QComboBox::down-arrow:disabled {{ image: url({{SETA_OFF}}); }}
QComboBox QAbstractItemView {{ background: {CARD}; border: 1px solid {LINE};
    selection-background-color: rgba(20,184,166,0.30); outline: none; padding: 4px; }}

QSlider::groove:horizontal {{ height: 6px; border-radius: 3px; background: {TEAL}; }}
QSlider::sub-page:horizontal {{ background: {AMBER}; border-radius: 3px; }}
QSlider::add-page:horizontal {{ background: {TEAL}; border-radius: 3px; }}
QSlider::handle:horizontal {{ background: {TXT}; width: 16px; height: 16px; margin: -5px 0; border-radius: 8px; }}
QLabel#teste {{ color: {AMBER}; font-size: 11px; font-weight: 800; }}
QLabel#treino {{ color: {TEAL}; font-size: 11px; font-weight: 800; }}

QPushButton {{ border-radius: 14px; padding: 11px 20px; font-size: 12px; font-weight: 800; }}
QPushButton#limpar {{ background: transparent; border: 1px solid {LINE}; color: {MUTED};
    padding: 8px 20px; font-size: 11px; border-radius: 14px; }}
QPushButton#limpar:hover {{ border-color: {TEAL}; color: {TEAL}; }}
QPushButton#executar {{ background: {TEAL}; color: {BG_TOPO}; font-size: 13px; border: 1px solid {TEAL}; }}
QPushButton#executar:hover {{ background: #2dd4bf; border-color: #2dd4bf; }}
QPushButton#todos {{ background: transparent; color: {TEAL}; font-size: 13px; border: 1px solid {TEAL}; }}
QPushButton#todos:hover {{ background: rgba(20,184,166,0.16); }}
QPushButton#executar:disabled, QPushButton#todos:disabled {{
    background: {CARD}; color: {DIM}; border: 1px solid {LINE}; }}

QFrame#cartao {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 14px; }}
QLabel#cartao_titulo {{ font-size: 14px; font-weight: 800; }}
QLabel#ponto {{ color: {TEAL}; font-size: 10px; }}
QFrame#tile {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 10px; }}
QLabel#metrica_nome {{ font-size: 10px; font-weight: 700; color: {MUTED}; }}
QLabel#metrica_valor {{ font-size: 22px; font-weight: 800; color: {TEAL}; }}
QProgressBar {{ background: {LINE}; border: none; border-radius: 3px; max-height: 6px; min-height: 6px; }}
QProgressBar::chunk {{ background: {TEAL}; border-radius: 3px; }}

QLabel#vazio_icone {{ color: {TEAL}; font-size: 26px; }}
QLabel#vazio_titulo {{ font-size: 18px; font-weight: 800; }}
QLabel#vazio_txt {{ font-size: 12px; color: {MUTED}; }}

QTableWidget {{ background: transparent; border: none; font-size: 12px; }}
QHeaderView::section {{ background: {PANEL}; color: {MUTED}; border: none;
    border-bottom: 1px solid {LINE}; padding: 6px; font-size: 10px; font-weight: 700; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; }}
QScrollBar::handle:vertical {{ background: {LINE}; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

QMessageBox {{ background: {PANEL}; }}
QMessageBox QLabel {{ color: {TXT}; font-size: 13px; }}
QMessageBox QPushButton {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 10px;
    padding: 7px 18px; min-width: 70px; font-size: 12px; }}
QMessageBox QPushButton:hover {{ border-color: {TEAL}; color: {TEAL}; }}
"""


def espacar(w, px=1.2):
    """Espaçamento entre letras (o QSS do Qt não suporta letter-spacing)."""
    f = w.font(); f.setLetterSpacing(QFont.AbsoluteSpacing, px); w.setFont(f)
    return w


def rotulo(texto, obj="rotulo", px=1.2):
    l = QLabel(texto); l.setObjectName(obj)
    return espacar(l, px)


def tabela(colunas, linhas, destaque_diag=False):
    t = QTableWidget(len(linhas), len(colunas))
    t.setHorizontalHeaderLabels([str(c) for c in colunas])
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QAbstractItemView.NoEditTriggers)
    t.setSelectionMode(QAbstractItemView.NoSelection)
    t.setFocusPolicy(Qt.NoFocus)
    t.setShowGrid(False)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    for i, linha in enumerate(linhas):
        for j, v in enumerate(linha):
            it = QTableWidgetItem(str(v))
            it.setTextAlignment(Qt.AlignCenter)
            if destaque_diag and i == j - 1:   # diagonal da matriz de confusão
                it.setForeground(QColor(TEAL))
                it.setBackground(QColor(20, 184, 166, 38))
                f = it.font(); f.setBold(True); it.setFont(f)
            t.setItem(i, j, it)
    t.horizontalHeader().setFixedHeight(32)
    t.setFixedHeight(32 + 34 * len(linhas) + 4)
    t.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    for i in range(len(linhas)):
        t.setRowHeight(i, 34)
    return t


def metrica(nome, valor):
    tile = QFrame(); tile.setObjectName("tile")
    box = QVBoxLayout(tile); box.setContentsMargins(14, 10, 14, 10); box.setSpacing(2)
    box.addWidget(rotulo(nome, "metrica_nome", 1.0))
    v = QLabel(valor); v.setObjectName("metrica_valor")
    box.addWidget(v)
    return tile


def barra(nome, frac):
    lay = QHBoxLayout(); lay.setSpacing(10)
    lay.addWidget(rotulo(nome, "dica", 1.0))
    b = QProgressBar(); b.setRange(0, 1000); b.setValue(int(max(0.0, min(1.0, frac)) * 1000))
    b.setTextVisible(False)
    lay.addWidget(b, 1)
    return lay


def cartao_resultado(r, tipo):
    c = QFrame(); c.setObjectName("cartao")
    lay = QVBoxLayout(c); lay.setContentsMargins(18, 16, 18, 16); lay.setSpacing(12)

    cab = QHBoxLayout(); cab.setSpacing(10)
    ponto = QLabel("●"); ponto.setObjectName("ponto")
    cab.addWidget(ponto)
    cab.addWidget(rotulo(r["metodo"].upper(), "cartao_titulo", 1.2))
    cab.addStretch(1)
    lay.addLayout(cab)

    linha = QHBoxLayout(); linha.setSpacing(10)
    if tipo == "classificacao":
        for nome, k in (("ACURÁCIA", "ac"), ("PRECISÃO", "pr"), ("RECALL", "re"), ("F1", "f1")):
            linha.addWidget(metrica(nome, f"{r[k]*100:.2f}%"))
    else:
        for nome, k in (("MAE", "mae"), ("MSE", "mse"), ("RMSE", "rmse"), ("R²", "r2")):
            linha.addWidget(metrica(nome, f"{r[k]:.4f}"))
    lay.addLayout(linha)
    lay.addLayout(barra("ACERTOS" if tipo == "classificacao" else "R²",
                        r["ac"] if tipo == "classificacao" else r["r2"]))

    if tipo == "classificacao":
        classes = r["classes"]
        lay.addWidget(rotulo("DETALHES POR CLASSE"))
        lay.addWidget(tabela(
            ["Classe", "Precisão", "Recall", "F1"],
            [[classes[i], f"{r['pr_cl'][i]:.3f}", f"{r['re_cl'][i]:.3f}", f"{r['f1_cl'][i]:.3f}"]
             for i in range(len(classes))]))
        lay.addWidget(rotulo("MATRIZ DE CONFUSÃO  ·  linhas: real  ·  colunas: previsto"))
        cm = r["cm"]
        lay.addWidget(tabela(
            [""] + classes,
            [[classes[i]] + [int(x) for x in cm[i]] for i in range(len(classes))],
            destaque_diag=True))
    return c


# ==============================================================
# ÁREA DE ARQUIVO (clique ou arraste)
# ==============================================================
class ZonaArquivo(QFrame):
    clicado = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("drop"); self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(64)
        lay = QHBoxLayout(self); lay.setContentsMargins(16, 0, 16, 0); lay.setSpacing(12)
        self.icone = QLabel("⇧"); self.icone.setObjectName("drop_icone")
        col = QVBoxLayout(); col.setSpacing(1)
        self.titulo = QLabel("Escolher arquivo .txt"); self.titulo.setObjectName("drop_titulo")
        self.sub = QLabel("clique aqui ou arraste para a janela"); self.sub.setObjectName("drop_sub")
        col.addStretch(1); col.addWidget(self.titulo); col.addWidget(self.sub); col.addStretch(1)
        lay.addWidget(self.icone); lay.addLayout(col, 1)
        for w in (self.icone, self.titulo, self.sub):       # o clique vai para o quadro
            w.setAttribute(Qt.WA_TransparentForMouseEvents)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicado.emit()

    def set_arquivo(self, nome=None):
        if nome:
            self.titulo.setText(nome); self.sub.setText("clique para trocar o arquivo")
        else:
            self.titulo.setText("Escolher arquivo .txt")
            self.sub.setText("clique aqui ou arraste para a janela")

    def set_arrasto(self, on):
        self.setProperty("arrasto", "true" if on else "false")
        self.style().unpolish(self); self.style().polish(self)


# ==============================================================
# JANELA PRINCIPAL
# ==============================================================
class Janela(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("raiz"); self.setAttribute(Qt.WA_StyledBackground, True)
        self.setAcceptDrops(True)
        self.setWindowTitle("ML Lab — Classificação e Regressão")
        self.resize(1280, 820)
        self.setMinimumSize(1000, 640)

        self.feat = self.alvo = self.caminho = None
        self.tipo = "classificacao"
        self.modulo = clas
        self.worker = None
        self.n_res = 0

        raiz = QVBoxLayout(self); raiz.setContentsMargins(0, 0, 0, 0); raiz.setSpacing(0)
        raiz.addWidget(self._topo())

        corpo = QHBoxLayout(); corpo.setContentsMargins(24, 22, 24, 24); corpo.setSpacing(20)
        corpo.addWidget(self._lateral())
        corpo.addWidget(self._painel_resultados(), 1)
        raiz.addLayout(corpo, 1)
        self.set_modo("classificacao")

    # ---------- topo ----------
    def _topo(self):
        topo = QWidget(); topo.setObjectName("topo"); topo.setFixedHeight(70)
        topo.setAttribute(Qt.WA_StyledBackground, True)
        l = QHBoxLayout(topo); l.setContentsMargins(28, 0, 28, 0); l.setSpacing(14)

        logo = QLabel("ML"); logo.setObjectName("logo"); logo.setFixedSize(40, 40)
        logo.setAlignment(Qt.AlignCenter)
        l.addWidget(logo)
        col = QVBoxLayout(); col.setSpacing(1)
        col.addStretch(1)
        col.addWidget(rotulo("ML LAB", "marca", 2.0))
        sub = QLabel("Classificação e regressão com scikit-learn"); sub.setObjectName("sub")
        col.addWidget(sub); col.addStretch(1)
        l.addLayout(col)
        l.addStretch(1)

        seg = QFrame(); seg.setObjectName("seg")
        sl = QHBoxLayout(seg); sl.setContentsMargins(4, 4, 4, 4); sl.setSpacing(4)
        self.grupo_modo = QButtonGroup(self); self.grupo_modo.setExclusive(True)
        self.bt_modo = {}
        for chave, texto in (("classificacao", "CLASSIFICAÇÃO"), ("regressao", "REGRESSÃO")):
            b = QPushButton(texto); b.setObjectName("aba"); b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor); espacar(b, 1.2)
            b.clicked.connect(lambda _=False, c=chave: self.set_modo(c))
            self.grupo_modo.addButton(b); self.bt_modo[chave] = b
            sl.addWidget(b)
        seg.setFixedHeight(40)
        l.addWidget(seg, 0, Qt.AlignVCenter)
        return topo

    # ---------- painel esquerdo ----------
    def _passo(self, n, texto):
        h = QHBoxLayout(); h.setSpacing(10)
        b = QLabel(str(n)); b.setObjectName("badge"); b.setFixedSize(22, 22)
        b.setAlignment(Qt.AlignCenter)
        h.addWidget(b); h.addWidget(rotulo(texto)); h.addStretch(1)
        return h

    def _lateral(self):
        f = QFrame(); f.setObjectName("painel"); f.setFixedWidth(380)
        l = QVBoxLayout(f); l.setContentsMargins(22, 22, 22, 22); l.setSpacing(10)

        l.addLayout(self._passo(1, "DADOS"))
        self.zona = ZonaArquivo(); self.zona.clicado.connect(self.carregar)
        l.addWidget(self.zona)

        stats = QHBoxLayout(); stats.setSpacing(10)
        self.st_valor = []
        self.st_nome3 = None
        for nome in ("INSTÂNCIAS", "ATRIBUTOS", "CLASSES"):
            card = QFrame(); card.setObjectName("stat"); card.setFixedHeight(54)
            cl = QVBoxLayout(card); cl.setContentsMargins(6, 6, 6, 6); cl.setSpacing(0)
            v = QLabel("—"); v.setObjectName("stat_valor"); v.setAlignment(Qt.AlignCenter)
            n = rotulo(nome, "stat_nome", 1.0); n.setAlignment(Qt.AlignCenter)
            cl.addWidget(v); cl.addWidget(n)
            stats.addWidget(card); self.st_valor.append(v)
            self.st_nome3 = n
        l.addLayout(stats)
        l.addSpacing(10)

        l.addLayout(self._passo(2, "MÉTODO"))
        self.cb_metodo = QComboBox()
        l.addWidget(self.cb_metodo)
        l.addSpacing(10)

        l.addLayout(self._passo(3, "DIVISÃO TREINO / TESTE"))
        self.sl_teste = QSlider(Qt.Horizontal); self.sl_teste.setRange(5, 95)
        self.sl_teste.setValue(20); self.sl_teste.setCursor(Qt.PointingHandCursor)
        l.addWidget(self.sl_teste)
        pcts = QHBoxLayout()
        self.lb_teste = QLabel(); self.lb_teste.setObjectName("teste")
        self.lb_treino = QLabel(); self.lb_treino.setObjectName("treino")
        pcts.addWidget(self.lb_teste); pcts.addStretch(1); pcts.addWidget(self.lb_treino)
        l.addLayout(pcts)
        self.sl_teste.valueChanged.connect(self._atualiza_pct)
        self._atualiza_pct(self.sl_teste.value())

        l.addStretch(1)

        self.bt_exec = QPushButton("▶  EXECUTAR"); self.bt_exec.setObjectName("executar")
        self.bt_todos = QPushButton("EXECUTAR TODOS"); self.bt_todos.setObjectName("todos")
        for b in (self.bt_exec, self.bt_todos):
            b.setCursor(Qt.PointingHandCursor); b.setEnabled(False); b.setMinimumHeight(46)
            espacar(b, 1.5)
            l.addWidget(b)
        self.bt_exec.clicked.connect(lambda: self.executar(False))
        self.bt_todos.clicked.connect(lambda: self.executar(True))
        return f

    def _atualiza_pct(self, v):
        self.lb_teste.setText(f"TESTE {v}%"); self.lb_treino.setText(f"TREINO {100 - v}%")

    # ---------- painel de resultados ----------
    def _painel_resultados(self):
        f = QFrame(); f.setObjectName("painel")
        l = QVBoxLayout(f); l.setContentsMargins(24, 22, 24, 22); l.setSpacing(14)

        cab = QHBoxLayout()
        tl = QVBoxLayout(); tl.setSpacing(0)
        tl.addWidget(rotulo("RESULTADOS", "titulo", 1.5))
        self.lbl_sub = QLabel("Nenhuma execução ainda"); self.lbl_sub.setObjectName("sub")
        tl.addWidget(self.lbl_sub)
        cab.addLayout(tl); cab.addStretch(1)
        self.bt_limpar = QPushButton("LIMPAR"); self.bt_limpar.setObjectName("limpar")
        self.bt_limpar.setCursor(Qt.PointingHandCursor); espacar(self.bt_limpar, 1.5)
        self.bt_limpar.clicked.connect(self.limpar)
        cab.addWidget(self.bt_limpar, 0, Qt.AlignTop)
        l.addLayout(cab)

        self.pilha = QStackedWidget()
        # página 0: estado vazio
        vazio = QWidget(); vl = QVBoxLayout(vazio); vl.setSpacing(6)
        vl.addStretch(1)
        ic = QLabel("◈"); ic.setObjectName("vazio_icone"); ic.setAlignment(Qt.AlignCenter)
        t1 = QLabel("Pronto para começar"); t1.setObjectName("vazio_titulo"); t1.setAlignment(Qt.AlignCenter)
        t2 = QLabel("1. Escolha CLASSIFICAÇÃO ou REGRESSÃO no topo  ·  2. Carregue um arquivo .txt\n"
                    "3. Escolha o método  ·  4. Clique em EXECUTAR")
        t2.setObjectName("vazio_txt"); t2.setAlignment(Qt.AlignCenter)
        for w in (ic, t1, t2):
            vl.addWidget(w)
        vl.addStretch(1)
        self.pilha.addWidget(vazio)
        # página 1: cartões
        self.area = QScrollArea(); self.area.setWidgetResizable(True)
        self.conteudo = QWidget(); self.conteudo.setStyleSheet("background: transparent;")
        self.col = QVBoxLayout(self.conteudo); self.col.setContentsMargins(0, 0, 6, 0); self.col.setSpacing(14)
        self.col.addStretch(1)
        self.area.setWidget(self.conteudo)
        self.pilha.addWidget(self.area)
        l.addWidget(self.pilha, 1)
        return f

    # ---------- arrastar e soltar ----------
    @staticmethod
    def _txt_da_url(e):
        for u in e.mimeData().urls():
            if u.isLocalFile() and u.toLocalFile().lower().endswith(".txt"):
                return u.toLocalFile()
        return None

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls() and self._txt_da_url(e):
            e.acceptProposedAction(); self.zona.set_arrasto(True)
        else:
            e.ignore()

    def dragLeaveEvent(self, e):
        self.zona.set_arrasto(False)

    def dropEvent(self, e):
        self.zona.set_arrasto(False)
        caminho = self._txt_da_url(e)
        if caminho:
            e.acceptProposedAction(); self.abrir(caminho)

    # ---------- ações ----------
    def set_modo(self, modo):
        """Chamado pelos botões CLASSIFICAÇÃO / REGRESSÃO do topo."""
        if self.worker and self.worker.isRunning():       # não troca durante execução
            self.bt_modo[self.tipo].setChecked(True)
            return
        self.tipo = modo
        self.modulo = clas if modo == "classificacao" else reg
        self.bt_modo[modo].setChecked(True)
        self.st_nome3.setText("CLASSES" if modo == "classificacao" else "ALVO")
        self.limpar()
        if self.caminho:                                  # relê o mesmo arquivo no novo modo
            self.abrir(self.caminho)
        else:
            self._sem_dados()

    def _sem_dados(self):
        self.feat = self.alvo = None
        self.zona.set_arquivo(None)
        for v in self.st_valor:
            v.setText("—")
        self.cb_metodo.clear(); self.cb_metodo.addItem("carregue um arquivo primeiro")
        self.cb_metodo.setEnabled(False)
        self.bt_exec.setEnabled(False); self.bt_todos.setEnabled(False)

    def carregar(self):
        caminho, _ = QFileDialog.getOpenFileName(self, "Abrir arquivo TXT", "", "Texto (*.txt);;Todos (*)")
        if caminho:
            self.abrir(caminho)

    def abrir(self, caminho):
        if self.worker and self.worker.isRunning():
            return
        try:
            feat, alvo = ler_txt(caminho, self.modulo, self.tipo)
        except Exception as e:
            QMessageBox.critical(self, "Erro ao ler o arquivo", str(e))
            self.caminho = None
            self._sem_dados()
            return
        self.feat, self.alvo, self.caminho = feat, alvo, caminho
        nome = caminho.replace("\\", "/").split("/")[-1]
        self.zona.set_arquivo(nome)
        self.st_valor[0].setText(str(len(self.alvo)))
        self.st_valor[1].setText(str(np.asarray(self.feat).shape[1]))
        self.st_valor[2].setText(str(len(set(self.alvo))) if self.tipo == "classificacao" else "numérico")
        self.cb_metodo.clear(); self.cb_metodo.addItems(list(self.modulo.METODOS))
        self.cb_metodo.setEnabled(True)
        self.bt_exec.setEnabled(True); self.bt_todos.setEnabled(True)
        self.limpar()

    def limpar(self):
        while self.col.count() > 1:                       # o último item é o espaçador
            w = self.col.takeAt(0).widget()
            if w:
                w.hide(); w.setParent(None); w.deleteLater()
        self.n_res = 0
        self.pilha.setCurrentIndex(0)
        self.lbl_sub.setText("Nenhuma execução ainda")

    def executar(self, todos):
        if self.feat is None or (self.worker and self.worker.isRunning()):
            return
        nomes = list(self.modulo.METODOS) if todos else [self.cb_metodo.currentText()]
        try:
            dados = self.modulo.preparar(self.feat, self.alvo, self.sl_teste.value())
        except Exception as e:
            QMessageBox.critical(self, "Erro", str(e)); return
        self.limpar(); self.pilha.setCurrentIndex(1)
        self.lbl_sub.setText("Executando…")
        self.bt_exec.setEnabled(False); self.bt_todos.setEnabled(False)
        self.bt_exec.setText("EXECUTANDO…")
        self.worker = Worker(self.modulo, nomes, dados)
        self.worker.resultado.connect(self.mostrar)
        self.worker.erro.connect(lambda m: QMessageBox.warning(self, "Erro no método", m))
        self.worker.finished.connect(self.fim)
        self.worker.start()

    def mostrar(self, r):
        self.col.insertWidget(self.col.count() - 1, cartao_resultado(r, self.tipo))
        self.n_res += 1
        self.lbl_sub.setText("Executando…")

    def fim(self):
        self.bt_exec.setEnabled(True); self.bt_todos.setEnabled(True)
        self.bt_exec.setText("▶  EXECUTAR")
        n = self.n_res
        self.lbl_sub.setText(f"{n} método{'s' if n != 1 else ''} executado{'s' if n != 1 else ''}"
                             if n else "Nenhuma execução ainda")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(montar_estilo())
    j = Janela()
    j.show()
    if len(sys.argv) > 1:          # opcional: python interface.py arquivo.txt
        j.abrir(sys.argv[1])
    sys.exit(app.exec())
