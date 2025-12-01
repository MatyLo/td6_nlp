from pathlib import Path
import mlflow
import numpy as np
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from time import sleep
import yaml

from src_rag import models

from FlagEmbedding import FlagModel

CONF = yaml.safe_load(open("config.yml"))

FOLDER = Path("data") / "raw" / "movies" / "wiki"
FILENAMES = [
    FOLDER / title for title in ["Inception.md", "The Dark Knight.md", "Deadpool.md", "Fight Club.md", "Pulp Fiction.md"]
]
DF = pd.read_csv("data/raw/movies/questions.csv", sep=";") 

ENCODER = SentenceTransformer('all-MiniLM-L6-v2')


def _load_ml_flow(conf):
    experiment_name = "RAG_Movies_2"
    mlruns_path = os.path.join(os.getcwd(), "mlruns")
    if not os.path.exists(mlruns_path):
        os.makedirs(mlruns_path)
    mlflow.set_tracking_uri(f"file://{mlruns_path}")
    mlflow.set_experiment(experiment_name)


_load_ml_flow(CONF)


def _generate_run_name(config):
    """Génère un nom descriptif pour l'expérimentation MLflow"""
    model_config = config.get("model", {})
    parts = []
    
    chunk_size = model_config.get("chunk_size", 256)
    parts.append(f"cs{chunk_size}")
    
    overlap = model_config.get("overlap", 0)
    if overlap > 0:
        parts.append(f"ov{overlap}")
    
    embedding = model_config.get("embedding_model", "bge-base")
    parts.append(f"emb_{embedding}")
    
    if model_config.get("small2big", False):
        context = model_config.get("small2big_context", 1)
        parts.append(f"s2b{context}")
    
    return "_".join(parts)


def run_evaluate_retrieval(config, rag=None):
    rag = rag or models.get_model(config)
    score = evaluate_retrieval(rag, FILENAMES, DF.dropna())

    description = str(config.get("model", {}))
    run_name = _generate_run_name(config)
    _push_mlflow_result(score, config, description, run_name)
    
    print(f"\n📊 Résultats: MRR={score['mrr']:.4f}, nb_chunks={score['nb_chunks']}")
    
    return rag


def run_evaluate_reply(config, rag=None):
    rag = rag or models.get_model(config)
    indexes = range(2, len(DF), 10)
    score = evaluate_reply(rag, FILENAMES, DF.iloc[indexes])

    description = str(config.get("model", {}))
    run_name = _generate_run_name(config) + "_reply"
    _push_mlflow_result(score, config, description, run_name)
    
    return rag


def _push_mlflow_result(score, config, description=None, run_name=None):
    with mlflow.start_run(description=description, run_name=run_name):
        df = score.pop("df_result")
        mlflow.log_table(df, artifact_file="df.json")
        mlflow.log_metrics(score)

        config_no_key = {
            key: val for key, val in config.items() if not key.endswith("_key")
        }

        mlflow.log_dict(config_no_key, "config.json")


def evaluate_reply(rag, filenames, df):
    rag.load_files(filenames)

    replies = []
    for question in tqdm(df["question"]):
        replies.append(rag.reply(question))
        sleep(2)

    df["reply"] = replies
    df["sim"] = df.apply(lambda row: calc_semantic_similarity(row["reply"], row["expected_reply"]), axis=1)
    df["is_correct"] = df["sim"] > .7

    return {
        "reply_similarity": df["sim"].mean(),
        "percent_correct": df["is_correct"].mean(),
        "df_result": df[["question", "reply", "expected_reply", "sim", "is_correct"]],
    }


def evaluate_retrieval(rag, filenames, df_question):
    rag.load_files(filenames)
    ranks = []
    for _, row in df_question.iterrows():
        chunks = rag._get_context(row.question)
        try:
            rank = next(i for i, c in enumerate(chunks) if row.text_answering in c)
        except StopIteration:
            rank = 0

        ranks.append(rank)
        
    df_question["rank"] = ranks
            
    mrr = np.mean([0 if r == 0 else 1 / r for r in ranks])

    return {
        "mrr": mrr,
        "nb_chunks": len(rag.get_chunks()),
        "df_result": df_question[["question", "text_answering", "rank"]],
    }


def calc_acceptable_chunks(chunks, text_to_find):
    acceptable_chunks = []
    for answer in text_to_find:
        chunks_ok = set(i for i, chunk in enumerate(chunks) if answer in chunk)
        acceptable_chunks.append(chunks_ok)

    return acceptable_chunks


def calc_mrr(sim_score, acceptable_chunks, top_n=5):
    ranks = []
    for this_score, this_acceptable_chunks in zip(sim_score, acceptable_chunks):
        indexes = reversed(np.argsort(this_score))
        try:
            rank = 1 + next(i for i, idx in enumerate(indexes) if idx in this_acceptable_chunks)
        except StopIteration:
            rank = len(this_score) + 1
        
        ranks.append(rank)
        
    return {
        "mrr": sum(1 / r if r < top_n + 1 else 0 for r in ranks) / len(ranks),
        "ranks": ranks,
    }


def calc_semantic_similarity(generated_answer: str, reference_answer: str) -> float:
    embeddings = ENCODER.encode([generated_answer, reference_answer])
    generated_embedding = embeddings[0].reshape(1, -1)
    reference_embedding = embeddings[1].reshape(1, -1)
    similarity = cosine_similarity(generated_embedding, reference_embedding)[0][0]
    return float(similarity)

   
if __name__ == "__main__":
    # --- EXPÉRIMENTATIONS CHUNK SIZE ET OVERLAP ---
    # model_config = {"chunk_size": 256}                    # mrr=0.19
    # model_config = {"chunk_size": 700, "overlap": 100}    # mrr=0.25, reply_sim=0.799
    # model_config = {"chunk_size": 512, "overlap": 50}     # mrr=0.009 (pas bon)
    # model_config = {"chunk_size": 256, "overlap": 75}     # mrr=0.24
    # model_config = {"chunk_size": 128, "overlap": 25}     # mrr=0.17
    # model_config = {"chunk_size": 1000}                   # mrr=0.24, reply_sim=0.747, percent_correct=0.66
    
    # model_config = {"chunk_size": 256, "overlap": 100}    # mrr=0.27, percent=0.66, simi=0.7161167992485894, nbchunks=500.0
    # model_config = {"chunk_size": 700}                    # mrr=0.28, percent=0.66, simi=0.7653752565383911, nbchunks=187.0
    # model_config = {"chunk_size": 512}                    # mrr=0.27, percent=0.88, simi=0.7998429238796234, nbchunks=220.0

    # --- EXPÉRIMENTATIONS EMBEDDINGS
    # model_config = {"chunk_size": 700, "embedding_model": "bge-base"}   # mrr=0.193
    # model_config = {"chunk_size": 1100, "embedding_model": "minilm"} #mrr=0.217
    # model_config = {"chunk_size": 700, "embedding_model": "e5-base"} # mrr=0.204
    # model_config = {"chunk_size": 1100, "embedding_model": "gte-base"} # mrr=0.227
    
    # --- EXPÉRIMENTATIONS SMALL2BIG
    # model_config = {"chunk_size": 1100, "small2big": True, "small2big_context": 1} # mrr=0.247
    model_config = {"chunk_size": 1100, "small2big": True, "small2big_context": 2} # mrr=197
    
    # model_config = {"chunk_size": 700, "embedding_model": "bge-base"} # mrr=0.193
    
    print(f"🔬 Config: {model_config}")
    # run_evaluate_retrieval({"model": model_config})
    run_evaluate_reply({"model": model_config}) 
