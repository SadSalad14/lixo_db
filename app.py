"""
Backend de denúncias de lixo com heatmap por clustering — versão para produção.

Diferenças da versão local:
  - Usa Postgres (via DATABASE_URL) em vez de SQLite, pra não perder dados
    quando o serviço reinicia no Render.
  - Denúncias agora guardam o tipo de lixo (usado pra sugerir o caminhão ideal).

Rotas:
  POST /denuncias       -> registra uma denúncia (lat, lng, tipo_lixo, descricao opcional)
  GET  /heatmap          -> retorna clusters (DBSCAN) com peso, prontos pro mapa
  GET  /pontos-coleta     -> retorna lista fixa de pontos de coleta/reciclagem

Rodar localmente para testar:
  pip install -r requirements.txt --break-system-packages
  export DATABASE_URL="postgresql://usuario:senha@host:5432/banco"
  python app.py

No Render: configure a variável de ambiente DATABASE_URL com a "Internal
Database URL" do banco Postgres criado lá, e defina o Start Command como:
  gunicorn app:app
"""

import os
import time

import numpy as np
import psycopg2
from flask import Flask, jsonify, request
from sklearn.cluster import DBSCAN

app = Flask(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL")

# --- distância aproximada: 1 grau de lat/lng ~ 111km. 300 metros ~ 0.0027 graus ---
EPS_GRAUS = 0.0027   # raio de agrupamento (~300m) — ajuste conforme necessidade
MIN_DENUNCIAS_POR_CLUSTER = 2

TIPOS_LIXO_VALIDOS = {"domestico", "entulho", "reciclavel", "volumoso", "perigoso"}

# Lista fixa de exemplo — troquem pelos pontos reais levantados pelo grupo
PONTOS_COLETA = [
    {"nome": "Ecoponto Boa Vista", "lat": -8.0578, "lng": -34.8829, "tipo": "ecoponto"},
    {"nome": "Cooperativa Recife Recicla", "lat": -8.0631, "lng": -34.8711, "tipo": "cooperativa"},
]


def get_conn():
    return psycopg2.connect(DATABASE_URL)


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS denuncias (
            id SERIAL PRIMARY KEY,
            lat DOUBLE PRECISION NOT NULL,
            lng DOUBLE PRECISION NOT NULL,
            tipo_lixo TEXT,
            descricao TEXT,
            criado_em DOUBLE PRECISION NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS pontos_coleta (
            id SERIAL PRIMARY KEY,
            nome TEXT NOT NULL,
            lat DOUBLE PRECISION NOT NULL,
            lng DOUBLE PRECISION NOT NULL,
            tipo TEXT,
            criado_em DOUBLE PRECISION NOT NULL
        )
    """)
    conn.commit()
    cur.close()
    conn.close()


@app.route("/denuncias", methods=["POST"])
def criar_denuncia():
    data = request.get_json(force=True)
    lat = data.get("lat")
    lng = data.get("lng")
    tipo_lixo = data.get("tipo_lixo")
    descricao = data.get("descricao", "")

    if lat is None or lng is None:
        return jsonify({"erro": "lat e lng são obrigatórios"}), 400

    if tipo_lixo is not None and tipo_lixo not in TIPOS_LIXO_VALIDOS:
        return jsonify({"erro": f"tipo_lixo deve ser um de {sorted(TIPOS_LIXO_VALIDOS)}"}), 400

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO denuncias (lat, lng, tipo_lixo, descricao, criado_em) VALUES (%s, %s, %s, %s, %s)",
        (lat, lng, tipo_lixo, descricao, time.time()),
    )
    conn.commit()
    cur.close()
    conn.close()
    return jsonify({"status": "ok"}), 201


# --- CAP_DENUNCIAS: quantidade de denúncias que já satura o peso em 1.0 (vermelho máximo) ---
CAP_DENUNCIAS = 10


def calcular_peso(num_denuncias):
    return min(1.0, num_denuncias / CAP_DENUNCIAS)


@app.route("/heatmap", methods=["GET"])
def heatmap():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT lat, lng FROM denuncias")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        return jsonify({"pontos": []})

    coords = np.array(rows)  # shape (n, 2) -> [lat, lng]

    # DBSCAN agrupa denúncias próximas; pontos isolados viram ruído (label -1)
    clustering = DBSCAN(eps=EPS_GRAUS, min_samples=MIN_DENUNCIAS_POR_CLUSTER).fit(coords)
    labels = clustering.labels_

    pontos = []

    # Denúncia isolada = cluster de tamanho 1. Mesma fórmula de peso dos clusters;
    # o app decide se mostra como círculo branco ou entra no heatmap colorido,
    # com base no campo num_denuncias (LIMITE_DESTAQUE no App.js).
    for (lat, lng), label in zip(coords, labels):
        if label == -1:
            pontos.append({"lat": lat, "lng": lng, "weight": calcular_peso(1), "num_denuncias": 1})

    for label in set(labels):
        if label == -1:
            continue
        cluster_pontos = coords[labels == label]
        centro_lat, centro_lng = cluster_pontos.mean(axis=0)
        pontos.append({"lat": centro_lat, "lng": centro_lng, "weight": calcular_peso(len(cluster_pontos)),
                        "num_denuncias": len(cluster_pontos)})

    return jsonify({"pontos": pontos})


@app.route("/pontos-coleta", methods=["GET"])
def listar_pontos_coleta():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT nome, lat, lng, tipo FROM pontos_coleta")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    informados_por_usuarios = [
        {"nome": nome, "lat": lat, "lng": lng, "tipo": tipo} for (nome, lat, lng, tipo) in rows
    ]

    return jsonify({"pontos": PONTOS_COLETA + informados_por_usuarios})


@app.route("/pontos-coleta", methods=["POST"])
def criar_ponto_coleta():
    data = request.get_json(force=True)
    nome = data.get("nome")
    lat = data.get("lat")
    lng = data.get("lng")
    tipo = data.get("tipo", "outro")

    if not nome or lat is None or lng is None:
        return jsonify({"erro": "nome, lat e lng são obrigatórios"}), 400

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO pontos_coleta (nome, lat, lng, tipo, criado_em) VALUES (%s, %s, %s, %s, %s)",
        (nome, lat, lng, tipo, time.time()),
    )
    conn.commit()
    cur.close()
    conn.close()
    return jsonify({"status": "ok"}), 201


init_db()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
