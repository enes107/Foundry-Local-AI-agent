import sqlite3
import json
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer


class LocalRAG:
    """
    Microsoft Foundry Local standartlarında SQLite ve 
    Vektör Embedding tabanlı yerel RAG motoru.
    """
    def __init__(self, db_path: str = "data/knowledge.db", model_name: str = "all-MiniLM-L6-v2"):
        self.db_path = db_path
        # Bilgisayarında zaten kurulu olan yerel model
        self.embedder = SentenceTransformer(model_name)
        self._init_db()

    def _init_db(self):
        """SQLite veritabanı tablosunu oluşturur."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT,
                    content TEXT,
                    embedding TEXT
                )
            """)
            conn.commit()

    def ingest_data_from_file(self, file_path: str = "data/knowledge.txt"):
        """knowledge.txt dosyasını okur, parçalar, embedding üretir ve SQLite veritabanına kaydeder."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except FileNotFoundError:
            print(f"Hata: {file_path} dosyasi bulunamadi.")
            return

        blocks = [b.strip() for b in content.split("\n\n") if b.strip()]

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM document_chunks")

            for block in blocks:
                lines = [l.strip() for l in block.split("\n") if l.strip()]
                if not lines:
                    continue
                topic = lines[0].replace("#", "").replace(":", "").strip()
                chunk_text = "\n".join(lines[1:]).strip() if len(lines) > 1 else block
                
                full_payload = f"{topic}: {chunk_text}"
                vector = self.embedder.encode(full_payload).tolist()

                cursor.execute("""
                    INSERT INTO document_chunks (topic, content, embedding)
                    VALUES (?, ?, ?)
                """, (topic, chunk_text, json.dumps(vector)))

            conn.commit()
            print(f"Veriler basariyla kaydedildi. Toplam parca: {len(blocks)}")

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """İki vektör arasındaki Cosine Similarity hesaplar."""
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return float(dot_product / (norm_v1 * norm_v2))

    def get_top_chunks(self, query: str, top_k: int = 2) -> List[Tuple[str, str, float]]:
        """Kullanıcı sorgusunu vektörleştirip en yakın SQLite kaydını bulur."""
        query_vector = self.embedder.encode(query)
        
        results = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT topic, content, embedding FROM document_chunks")
            rows = cursor.fetchall()
            
            for topic, content, emb_json in rows:
                doc_vector = np.array(json.loads(emb_json))
                similarity = self._cosine_similarity(query_vector, doc_vector)
                results.append((topic, content, similarity))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]