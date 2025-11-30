import os
from dotenv import load_dotenv
import requests
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
# from langgraph.prebuilt import create_react_agent

load_dotenv()

@tool
def get_weather(city: str) -> str:
    """Hakee nykyisen sään annetulle kaupungille."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric&lang=fi"
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        temp = data['main']['temp']
        desc = data['weather'][0]['description']
        return f"{city}: {temp}°C, {desc}"
    return f"En löytänyt säätietoja kaupungille {city}"

def main():
    llm = ChatOllama(
        model="qwen3:8b",
        #model="llama3.2:3b",
        #model="qwen2.5:7b",
        temperature=0,
    )
    
    tools = [get_weather]
    
    agent = create_agent(
            llm, 
            tools,
            system_prompt="You are a helpful assistant. Only use tools when explicitly needed. For general conversation, just respond normally without using any tools."
    )

    print("Weather Agent - Kysy mitä vain! (kirjoita 'exit' lopettaaksesi)\n")
   
    messages = []

    while True:
        user_input = input("Sinä: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        try:
            messages.append({
                "role": "user", "content": user_input})

            result = agent.invoke({
                "messages": messages
            })

            ai_messages = [m for m in result["messages"] if hasattr(m, 'type') and m.type == 'ai' and m.content]
            if ai_messages:
                last_message = ai_messages[-1]
                print(f"\nAgentti: {last_message.content}\n")
                messages.append({"role": "assistant", "content": last_message.content})

        except Exception as e:
            print(f"Virhe: {e}\n")

if __name__ == "__main__":
    main()

