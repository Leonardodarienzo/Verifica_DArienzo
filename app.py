import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import json


st.set_page_config(page_title="AI Chef", layout="wide", page_icon="👨‍🍳")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "dispensa" not in st.session_state:
    st.session_state.dispensa = [] # Lista di dizionari: {"item": "", "qty": "", "expiry": ""}
if "preferenze" not in st.session_state:
    st.session_state.preferenze = []


def update_kitchen_state(text, api_key):
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=api_key)
    extract_prompt = ChatPromptTemplate.from_template("""
    Analizza il messaggio dell'utente e identifica ingredienti o preferenze alimentari.
    Estrai in JSON:
    - ingredients: lista di oggetti {{"item": "nome", "qty": "quantita", "expiry": "vicina/lontana/null"}}
    - preferences: lista di stringhe (es. "no uova", "diabetico")
    
    Messaggio: {input}
    Rispondi SOLO con il JSON.
    """)
    chain = extract_prompt | llm
    try:
        response = chain.invoke({"input": text})
        data = json.loads(response.content.strip().replace('```json', '').replace('```', ''))
        
        
        for ing in data.get("ingredients", []):
            st.session_state.dispensa.append(ing)
        
        for pref in data.get("preferences", []):
            if pref not in st.session_state.preferenze:
                st.session_state.preferenze.append(pref)
    except:
        pass

with st.sidebar:
    st.header("🛒 Dispensa Virtuale")
    if not st.session_state.dispensa:
        st.write("La dispensa è vuota.")
    else:
        for i, ing in enumerate(st.session_state.dispensa):
            st.success(f"**{ing['item']}** ({ing['qty']}) - Scadenza: {ing['expiry']}")
    
    st.divider()
    st.header("Preferenze & Vincoli")
    for pref in st.session_state.preferenze:
        st.warning(pref)
    
    st.divider()
    api_key = st.text_input("Inserisci Groq API Key", type="password")
    if st.button("Reset Totale"):
        st.session_state.chat_history = []
        st.session_state.dispensa = []
        st.session_state.preferenze = []
        st.rerun()


st.title("👨‍🍳 Ai chef")
st.info("Dimmi cosa hai in frigo. Ti proporrò delle ricette adatte quando avrò abbastanza informazioni!")

for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

user_input = st.chat_input("Es: Ho due uova e della farina, ma non posso mangiare zuccheri...")

if user_input:
    if not api_key:
        st.error("Inserisci la chiave API nella sidebar!")
    else:
        
        update_kitchen_state(user_input, api_key)
        

        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append(HumanMessage(content=user_input))

        
        try:
            with st.spinner("Lo Chef sta pensando..."):
                llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=api_key)
                
            
                dispensa_txt = ""
                for ing in st.session_state.dispensa:
                    dispensa_txt += f"- {ing['item']} (Quantità: {ing['qty']}, Scadenza: {ing['expiry']})\n"
                
                sufficiente = "SI" if len(st.session_state.dispensa) >= 3 else "NO"
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", f"""Sei un assistente culinario esperto. 
                    STATO ATTUALE DISPENSA:
                    {dispensa_txt if dispensa_txt else 'La dispensa è vuota.'}
                    
                    PREFERENZE: {", ".join(st.session_state.preferenze) if st.session_state.preferenze else 'Nessuna'}
                    INFORMAZIONI SUFFICIENTI PER RICETTE: {sufficiente}.
                    
                    REGOLE:
                    1. Se INFORMAZIONI SUFFICIENTI = NO: Non dare ricette. Fai domande mirate.
                    2. Se INFORMAZIONI SUFFICIENTI = SI: Proponi 3 ricette complete (Nome, Tempo, Ingredienti, Preparazione).
                    3. Dai priorità agli ingredienti vicini alla scadenza.
                    4. Rispondi in italiano."""),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{input}")
                ])
                
                chain = prompt | llm
                response = chain.invoke({
                    "input": user_input,
                    "history": st.session_state.chat_history[:-1]
                })
                
                with st.chat_message("assistant"):
                    st.markdown(response.content)
                st.session_state.chat_history.append(AIMessage(content=response.content))
                
                
                st.rerun()
        except Exception as e:
            st.error(f"Errore: {e}")