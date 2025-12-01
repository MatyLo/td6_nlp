# Rapport d'Expérimentations RAG

Approches Expérimentés : 
- Chunk Size & Overlap
- Small-to-Big
- Modèles d'Embeddings

## Métriques Utilisées

| Métrique | Description |
|----------|-------------|
| **MRR** | Mean Reciprocal Rank - mesure la qualité du ranking des documents retrouvés |
| **Percent** | Pourcentage de réponses correctes |
| **Similarity** | Similarité cosinus entre la réponse générée et la référence |
| **Nb Chunks** | Nombre de chunks générés (impact sur la performance) |


## 1. Expérimentations : Chunk Size & Overlap

Objectif : Trouver la taille de chunk et le chevauchement optimaux.

| Chunk Size | Overlap | MRR | Autres métriques |
|------------|---------|-----|------------------|
| 256 | - | 0.19 | - |
| 700 | 100 | 0.25 | similarity=0.799 |
| 512 | 50 | 0.009 | 
| 256 | 75 | 0.24 | - |
| 128 | 25 | 0.17 | - |
| 1000 | - | 0.24 | similarity=0.747, percent=0.66 |
| **256** | **100** | **0.27** | percent=0.66, similarity=0.72, chunks=500 |
| **700** | **-** | **0.28** | percent=0.66, similarity=0.77, chunks=187 |
| **512** | **-** | **0.27** | percent=0.88, similarity=0.80, chunks=220 |

**Observations :**
- Les chunks de taille 700 offrent le meilleur MRR (0.28) avec un bon équilibre chunks/performance
- La taille 512 atteint le meilleur taux de réponses correctes (88%)
- Un overlap trop grand peut dégrader les performances (ex: 512 + overlap 50)


## 2. Expérimentations : Modèles d'Embeddings


| Modèle | Chunk Size | MRR |
|--------|------------|-----|
| Default (all-MiniLM-L6-v2) | 700 | 0.28 |
| bge-base | 700 | 0.193 |
| minilm | 1100 | 0.217 |
| e5-base | 700 | 0.204 |
| gte-base | 1100 | 0.227 |

**Observations :**
- Le modèle par défaut (all-MiniLM-L6-v2) reste le plus performant
- Les modèles alternatifs n'apportent pas d'amélioration significative sur notre corpus

---

## 3. Expérimentations : Small-to-Big Retrieval

Objectif : Tester la technique Small-to-Big qui indexe de petits chunks mais retourne le contexte élargi.

| Chunk Size | Small2Big Context | MRR |
|------------|-------------------|-----|
| 1100 | 1 | 0.247 |
| 1100 | 2 | 0.197 |

**Observations :**
- Small2Big avec contexte=1 améliore légèrement les résultats
- Un contexte trop large (2) dégrade les performances

---

## Configuration Optimale Retenue (selon MRR)

```python
model_config = {
    "chunk_size": 700,
    "overlap": 0
}
# MRR=0.28, percent=0.66, similarity=0.77
```


## Conclusions

1. **Chunk Size** : Une taille entre 500-700 tokens offre le meilleur compromis
2. **Overlap** : Un overlap modéré (≤100) peut aider, mais pas systématiquement
3. **Embeddings** : Le modèle all-MiniLM-L6-v2 reste optimal pour ce cas d'usage
4. **Small2Big** : Apporte un gain marginal avec un contexte de 1


