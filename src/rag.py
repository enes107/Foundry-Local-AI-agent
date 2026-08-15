import os
from typing import List, Tuple

class SimpleRAG:
    """
    Belgeleri okuyan ve kullanici sorusuna en uygun metin parcalarini
    (context) getiren hafif RAG motoru.
    """
    def __init__(self, data_folder: str = "data"):
        self.data_folder = data_folder
        self.documents: List[str] = []
        self.load_documents()

    def load_documents(self):
        """data klasorundeki tum txt dosyalarini okur."""
        self.documents = []
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder, exist_ok=True)
            return

        for filename in os.listdir(self.data_folder):
            if filename.endswith(".txt"):
                filepath = os.path.join(self.data_folder, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:
                            self.documents.append(content)
                except Exception as e:
                    print(f"[RAG Hatasi] Dosya okunamadi: {filename} -> {e}")

    def search(self, query: str) -> Tuple[str, List[str]]:
        """
        Soru icerisindeki kelimelere gore belgelerden en alakali bolumu ceker.
        """
        if not self.documents or not query.strip():
            return "", []

        query_words = set(query.lower().split())
        matched_chunks = []
        sources = []

        for doc in self.documents:
            # Belgeyi paragraflara ayirarak ara
            paragraphs = doc.split("\n\n")
            for para in paragraphs:
                para_words = set(para.lower().split())
                # Soru ile paragraf arasindaki ortak kelime kesisimi
                common = query_words.intersection(para_words)
                if common:
                    matched_chunks.append((len(common), para))

        if not matched_chunks:
            return "", []

        # En cok kelime eslesen paragraflari sirala
        matched_chunks.sort(key=lambda x: x[0], reverse=True)
        top_context = matched_chunks[0][1]
        sources.append("data/knowledge.txt")

        return top_context, sources