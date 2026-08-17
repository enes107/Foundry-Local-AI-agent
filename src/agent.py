import os
from openai import OpenAI
from src.schemas import AgentResponse
from src.rag import LocalRAG

class LocalAgent:
    def __init__(self, agent_name: str = "Foundry Local HR & Tech Support Agent", db_path: str = "data/knowledge.db"):
        self.agent_name = agent_name
        self.rag = LocalRAG(db_path=db_path)
        self.processed_count = 0
        
        # Foundry Local OpenAI-Uyumlu Local Endpoint
        try:
            self.client = OpenAI(
                base_url= "http://127.0.0.1:56348/v1",
                api_key="foundry-local"
            )
            self.model_alias = "qwen2.5-coder-0.5b-instruct-generic-cpu:4"
            self.llm_ready = True
        except Exception as e:
            print(f"[UYARI] LLM istemcisi başlatılamadı: {e}")
            self.llm_ready = False

    def _generate_llm_response(self, user_query: str, context: str) -> str:
        system_prompt = (
            "Sen şirket içi yardım asistanısın. "
            "Soruya sadece verilen metne dayanarak 1-2 cümlelik kısa ve doğrudan bir Türkçe yanıt ver. "
            "Soru metnini yanıtta tekrarlama."
        )
        
        user_prompt = f"Metin:\n{context}\n\nSoru: {user_query}\nYanıt:"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_alias,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=80,             # Uzun döngülere girmesini engeller
                frequency_penalty=1.2,     # Cümle tekrarlarını cezalandırır
                stop=["\nSen:", "\nSoru:", "\n1.", "\n-", "###"] # Liste veya yeni soru başlatmasını keser
            )
            reply = response.choices[0].message.content.strip()
            # Eğer model yine de boş veya çok bozuk dönerse ham bağlamı ver
            return reply if len(reply) > 5 else context
        except Exception:
            return context

    def process_query(self, user_query: str) -> AgentResponse:
        self.processed_count += 1
        
        # 1. RAG ile SQLite'tan en alakalı parçayı getir (threshold: 0.60)
        chunks = self.rag.get_top_chunks(user_query, top_k=2)
        
        if chunks and chunks[0][2] >= 0.55:
            topic, content, similarity = chunks[0]
            confidence = float(similarity)
            sources = [topic]
            raw_context = content
            
            
            reply = raw_context
        else:
            # Eşik altı (Out-of-scope) Fallback mekanizması
            confidence = float(chunks[0][2]) if chunks else 0.0
            sources = []
            reply = (
                "Üzgünüm, şirket bilgi tabanında bu konuyla ilgili doğrulanmış bir bilgi bulunamadı. "
                "Yalnızca çalışma saatleri, fazla mesai, hibrit çalışma, maaş ve bilgi güvenliği konularında yardımcı olabilirim."
            )
            
        return AgentResponse(
            agent_name=self.agent_name,
            query=user_query,
            reply=reply,
            sources=sources,
            confidence_score=confidence
        )

    def get_processed_count(self) -> int:
        return self.processed_count