import numpy as np
import re
import tiktoken
import openai
import yaml

from FlagEmbedding import FlagModel
from sentence_transformers import SentenceTransformer

CONF = yaml.safe_load(open("config.yml"))

CLIENT = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=CONF["groq_key"],
)

tokenizer = tiktoken.get_encoding("cl100k_base")

EMBEDDING_MODELS = {
    "bge-base": {"type": "flag", "name": "BAAI/bge-base-en-v1.5"},
    "minilm": {"type": "sentence_transformer", "name": "all-MiniLM-L6-v2"},
    "e5-base": {"type": "sentence_transformer", "name": "intfloat/e5-base-v2"},
    "gte-base": {"type": "sentence_transformer", "name": "thenlper/gte-base"},
}


def get_model(config):
    if config:
        return RAG(**config.get("model", {}))
    else:
        return RAG()


class RAG:
    def __init__(
        self,
        chunk_size=256,
        overlap=0,
        embedding_model="bge-base",
        top_k=5,
        small2big=False,
        small2big_context=1,
    ):
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._embedding_model = embedding_model
        self._top_k = top_k
        self._small2big = small2big
        self._small2big_context = small2big_context
        
        self._embedder = None
        self._loaded_files = set()
        self._texts = []
        self._chunks = []
        self._corpus_embedding = None
        self._client = CLIENT

    def load_files(self, filenames):
        texts = []
        for filename in filenames:
            if filename in self._loaded_files:
                continue

            with open(filename) as f:
                texts.append(f.read())
                self._loaded_files.add(filename)

        self._texts += texts
        chunks_added = self._compute_chunks(texts)
        self._chunks += chunks_added

        new_embedding = self.embed_corpus(chunks_added)
        if self._corpus_embedding is not None:
            self._corpus_embedding = np.vstack([self._corpus_embedding, new_embedding])
        else:
            self._corpus_embedding = new_embedding

    def get_corpus_embedding(self):
        return self._corpus_embedding

    def get_chunks(self):
        return self._chunks

    def embed_questions(self, questions):
        embedder = self.get_embedder()
        model_config = EMBEDDING_MODELS.get(self._embedding_model, EMBEDDING_MODELS["bge-base"])
        
        if model_config["type"] == "flag":
            return embedder.encode_queries(questions)
        else:
            if "e5" in self._embedding_model:
                questions = ["query: " + q for q in questions]
            return embedder.encode(questions)

    def _compute_chunks(self, texts):
        return sum(
            (chunk_markdown(txt, chunk_size=self._chunk_size, overlap=self._overlap) for txt in texts),
            [],
        )

    def embed_corpus(self, chunks):
        embedder = self.get_embedder()
        model_config = EMBEDDING_MODELS.get(self._embedding_model, EMBEDDING_MODELS["bge-base"])
        
        if model_config["type"] == "flag":
            return embedder.encode(chunks)
        else:
            if "e5" in self._embedding_model:
                chunks = ["passage: " + c for c in chunks]
            return embedder.encode(chunks)

    def get_embedder(self):
        if not self._embedder:
            model_config = EMBEDDING_MODELS.get(self._embedding_model, EMBEDDING_MODELS["bge-base"])
            
            if model_config["type"] == "flag":
                self._embedder = FlagModel(
                    model_config["name"],
                    query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                    use_fp16=True,
                    device="cpu"
                )
            else:
                self._embedder = SentenceTransformer(model_config["name"])
        
        return self._embedder

    def reply(self, query):
        prompt = self._build_prompt(query)
        res = self._client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="openai/gpt-oss-20b",
        )
        return res.choices[0].message.content

    def _build_prompt(self, query):
        context_str = "\n".join(self._get_context(query))

        return f"""Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
If the answer is not in the context information, reply \"I cannot answer that question\".
Query: {query}
Answer:"""

    def _get_context(self, query):
        query_embedding = self.embed_questions([query])
        sim_scores = query_embedding @ self._corpus_embedding.T
        indexes = list(np.argsort(sim_scores[0]))[-self._top_k:][::-1]
        
        # Small2Big: étendre les chunks avec le contexte environnant
        if self._small2big:
            extended_chunks = []
            for idx in indexes:
                chunk_text = self._get_extended_chunk(idx)
                extended_chunks.append(chunk_text)
            return extended_chunks
        
        return [self._chunks[i] for i in indexes]

    def _get_extended_chunk(self, chunk_idx):
        """Récupère le chunk avec son contexte (small2big)"""
        context_size = self._small2big_context
        start_idx = max(0, chunk_idx - context_size)
        end_idx = min(len(self._chunks), chunk_idx + context_size + 1)
        extended_chunks = self._chunks[start_idx:end_idx]
        return "\n".join(extended_chunks)


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


def parse_markdown_sections(md_text: str) -> list[dict[str, str]]:
    pattern = re.compile(r"^(#{1,6})\s*(.+)$")
    lines = md_text.splitlines()

    sections = []
    header_stack = []
    current_section = {"headers": [], "content": ""}

    for line in lines:
        match = pattern.match(line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()

            if current_section["content"]:
                sections.append(current_section)

            header_stack = header_stack[:level - 1]
            header_stack.append(title)

            current_section = {
                "headers": header_stack.copy(),
                "content": ""
            }
        else:
            current_section["content"] += line + "\n"

    if current_section["content"]:
        sections.append(current_section)

    return sections


def chunk_markdown(md_text: str, chunk_size: int = 128, overlap: int = 0) -> list[dict]:
    parsed_sections = parse_markdown_sections(md_text)
    chunks = []

    for section in parsed_sections:
        tokens = tokenizer.encode(section["content"])
        i = 0
        while i < len(tokens):
            token_chunk = tokens[i:i + chunk_size]
            if token_chunk:
                chunks.append(tokenizer.decode(token_chunk))
            i += chunk_size - overlap
            if i < 0:
                i = 0
    return chunks
