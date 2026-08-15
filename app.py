import os
from dotenv import load_dotenv
from src.agent import LocalAgent

load_dotenv()

def main():
    print("==================================================")
    print("      FOUNDRY LOCAL AI AGENT - CANLI TERMINAL     ")
    print("==================================================")
    print("Cikmak icin 'q', 'exit' veya 'cikis' yazabilirsiniz.\n")

    bot = LocalAgent(agent_name="FoundryBot")
    print("-" * 50)

    while True:
        try:
            user_input = input("\nSen: ")
            
            if user_input.strip().lower() in ["q", "exit", "cikis"]:
                print(f"\n[{bot.agent_name}] Oturum sonlandirildi. Toplam islenen mesaj: {bot.get_history_summary()}")
                print("Gorusmek uzere!")
                break

            if not user_input.strip():
                continue

            # Pydantic AgentResponse nesnesi doner
            response = bot.process_query(user_input)
            print(f"[{response.agent_name}]: {response.reply}")

        except KeyboardInterrupt:
            print("\n\nProgram kullanici tarafindan durduruldu.")
            break

if __name__ == "__main__":
    main()