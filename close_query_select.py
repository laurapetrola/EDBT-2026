import json
import faiss
from sentence_transformers import SentenceTransformer

class SQLQueryRetriever:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.pares = []
        self.index = None

    def carregar_json(self, caminho_arquivo):
        """
        Lê o arquivo JSON e armazena os pares (pergunta, SQL) na estrutura interna.
        """
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            dados = json.load(f)

        self.pares = []
        for item in dados:
            for pergunta, sql in item.items():
                self.pares.append((pergunta, sql.replace("'", '"')))

    def construir_index(self):
        """
        Constrói o índice FAISS com embeddings das perguntas.
        """
        perguntas = [pergunta for pergunta, _ in self.pares]
        embeddings = self.model.encode(perguntas, convert_to_numpy=True)

        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def buscar(self, query, k=2):
        """
        Busca os k pares mais similares à query.
        """
        if self.index is None:
            raise ValueError("O índice ainda não foi construído. Use construir_index().")

        query_embedding = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(query_embedding, k)
        resultados = [self.pares[i] for i in I[0]]
        return resultados, D

    def buscar_filtrado(self, query, top_k=2, threshold=0.5):
        """
        Busca os top_k pares mais similares à query e filtra pela distância (threshold).
        """
        resultados, distancias = self.buscar(query, k=top_k)
        filtrados = [(q, sql) for (q, sql), d in zip(resultados, distancias[0]) if d < threshold]
        return filtrados