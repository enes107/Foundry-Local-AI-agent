import os
import requests
from typing import List
from src.schemas import ChatMessage, AgentResponse
from src.rag import SimpleRAG

class LocalAgent:
    """
    Foundry Local projesi icin Yapay Zeka Ajan sinifi.
    Bellek yonetimi, Pydantic dogrulamasi, RAG motoru ve LLM isteklerini yonetir.
    """
    def __init__(self, agent_name: str = "FoundryBot"):
        self.agent_name = agent_name
        self.system_prompt = os.getenv("SYSTEM_PROMPT", "Sen yardimsever bir yerel yapay zeka asistanisin.")
        self.local_endpoint = os.getenv("LOCAL_LLM_ENDPOINT", "http://localhost:11434/api/generate")
        self.history: List[ChatMessage] = []
        
        # RAG motorunu baslat
        self.rag = SimpleRAG(data_folder="data")
        
        # Sistem talimatini gecmise ekle
        self.history.append(ChatMessage(role="system", content=self.system_prompt))
        print(f"[{self.agent_name}] Ajan ve RAG Bilgi Bankasi hazirlandi.")

    def process_query(self, user_query: str) -> AgentResponse:
        clean_query = user_query.strip()
        if not clean_query:
            return AgentResponse(
                agent_name=self.agent_name,
                reply="Lutfen bos bir mesaj gondermeyin.",
                status_code=400
            )

        # Kullanici mesajini kaydet
        self.history.append(ChatMessage(role="user", content=clean_query))

        # 1. RAG ile Bilgi Bankasini Tara
        context, sources = self.rag.search(clean_query)

        # 2. Yanit Uretimi
        reply_text = self._generate_answer(clean_query, context)

        # Asistan yanitini kaydet
        self.history.append(ChatMessage(role="assistant", content=reply_text))

        return AgentResponse(
            agent_name=self.agent_name,
            reply=reply_text,
            status_code=200,
            sources=sources
        )

    def _generate_answer(self, prompt: str, context: str) -> str:
        if context:
            return (
                f"[RAG Bilgi Bankasından Yanıt]:\n"
                f"{context}\n\n"
                f"(Kaynak: data/knowledge.txt | Gecmis Mesaj: {len(self.history)})"
            )

        # RAG'de bulunamazsa genel yanit mekanizmasi
        return f"[Yerel Model Yanıtı]: '{prompt}' sorusu islendi. (Ek bir yerel dokuman bilgisi bulunamadi)"

    def get_history_summary(self) -> int:
        return len(self.history)