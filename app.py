import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import json
import re

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="AI Kitchen Pro - Token Monitor", layout="wide", page_icon="👨‍🍳")

# --- COSTANTI DI LIMITAZIONE ---
MAX_SESSION_TOKENS = 30000  # Soglia massima per evitare blocchi nel piano free

# 1. INIZIALIZZAZIONE STATO
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "dispensa" not in st.session_state:
    st.session_state.dispensa = [] 
if "preferenze" not in st.session_state:
    st.session_state.preferenze = []
if "num_persone" not in st.session_state:
    st.session_state.num_persone = "Non specificato"
if "ultimo_giudizio" not in st.session_state:
    st.session_state.ultimo_giudizio = "In attesa..."
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0

# --- FUNZIONE ESTRAZIONE DATI CON MONITORAGGIO TOKEN ---
def update_kitchen_state(text, api_key):
    try:
        llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=api_key)
        extract_prompt = ChatPromptTemplate.from_template("""
        Estrai dati dal messaggio per la cucina. 
        Rispondi SOLO con JSON:
        {{
            "ingredients": [{{ "item": "nome", "qty": "quantita", "expiry": "scadenza o null" }}],
            "preferences": ["stringa"],
            "people": "numero o null"
        }}
        Messaggio: {input}
        """)
        chain = extract_prompt | llm
        response = chain.invoke({"input": text})
        
        # Monitoraggio Token (Pag. 255)
        if hasattr(response, 'response_metadata'):
            usage = response.response_metadata.get('token_usage', {})
            st.session_state.total_tokens += usage.get('total_tokens', 0)

        data = json.loads(response.content.strip().replace('```json', '').replace('```', ''))
        if data.get("people"): st.session_state.num_persone = str(data["people"])
        for new_ing in data.get("ingredients", []):
            nome = new_ing['item'].lower().strip()
            trovato = False
            for old in st.session_state.dispensa:
                if old['item'].lower().strip() == nome:
                    if new_ing['qty'] != "null": old['qty'] = str(new_ing['qty'])
                    if new_ing['expiry'] != "null": old['expiry'] = str(new_ing['expiry'])
                    trovato = True
                    break
            if not trovato: st.session_state.dispensa.append(new_ing)
        for p in data.get("preferences", []):
            if p.lower() not in [x.lower() for x in st.session_state.preferenze]:
                st.session_state.preferenze.append(p)
    except Exception as e:
        print(f"Errore estrazione: {e}")

# --- SIDEBAR (MONITORAGGIO TOKEN E STATO) ---
with st.sidebar:
    st.header("📊 Monitor Risorse")
    
    # Visualizzazione Token
    token_perc = min(st.session_state.total_tokens / MAX_SESSION_TOKENS, 1.0)
    st.write(f"Token utilizzati: **{st.session_state.total_tokens}** / {MAX_SESSION_TOKENS}")
    st.progress(token_perc)
    
    if st.session_state.total_tokens > MAX_SESSION_TOKENS * 0.9:
        st.warning("⚠️ Attenzione: sei vicino al limite token della sessione.")

    st.divider()
    st.header("🛒 Stato Cucina")
    st.info(f"👥 Persone: {st.session_state.num_persone}")
    
    for ing in st.session_state.dispensa:
        st.success(f"**{ing['item']}**\n{ing['qty']} | Scad: {ing['expiry']}")
    
    for pref in st.session_state.preferenze:
        st.error(pref)
    
    st.divider()
    api_key = st.text_input("Groq API Key", type="password")
    if st.button("🔄 Reset Totale"):
        st.session_state.clear()
        st.rerun()

# --- LAYOUT PRINCIPALE ---
col_chat, col_judge = st.columns([0.7, 0.3])

with col_chat:
    st.title("👨‍🍳 AI Multi-Agent Kitchen")
    for message in st.session_state.chat_history:
        avatar = "👨‍🍳" if message["role"] == "assistant" else None
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # Disabilita input se token esauriti
    if st.session_state.total_tokens < MAX_SESSION_TOKENS:
        user_input = st.chat_input("Inserisci dati...")
    else:
        st.error("🛑 Limite token raggiunto per questa sessione. Reset dell'app necessario.")
        user_input = None

with col_judge:
    st.header("⚖️ Giudice Critico")
    st.warning(st.session_state.ultimo_giudizio)

# --- LOGICA CORE ---
if user_input:
    with col_chat:
        if not api_key:
            st.error("Inserisci l'API Key!")
        else:
            update_kitchen_state(user_input, api_key)
            st.chat_message("user").markdown(user_input)
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            try:
                llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=api_key)
                sufficiente = "SI" if (len(st.session_state.dispensa) >= 5 and st.session_state.num_persone != "Non specificato") else "NO"
                
                # CHEF
                with st.spinner("Chef al lavoro..."):
                    disp_txt = "\n".join([f"- {i['item']} ({i['qty']}, scad: {i['expiry']})" for i in st.session_state.dispensa])
                    chef_prompt = ChatPromptTemplate.from_messages([
                        ("system", f"Sei uno Chef. Dati: {disp_txt} | Persone: {st.session_state.num_persone}. REGOLE: Se SOGLIA={sufficiente} e' SI, dai 3 ricette complete con dosi reali e svolgimento. Rispetta vincoli: {st.session_state.preferenze}. Rispondi in italiano."),
                        MessagesPlaceholder(variable_name="history"),
                        ("human", "{input}")
                    ])
                    history_lc = [HumanMessage(content=m["content"]) if m["role"]=="user" else AIMessage(content=m["content"]) for m in st.session_state.chat_history[:-1]]
                    chef_res = (chef_prompt | llm).invoke({"input": user_input, "history": history_lc})
                    
                    # Track Tokens
                    st.session_state.total_tokens += chef_res.response_metadata.get('token_usage', {}).get('total_tokens', 0)

                # GIUDICE
                if sufficiente == "SI":
                    with st.spinner("Giudice al lavoro..."):
                        judge_prompt = ChatPromptTemplate.from_template("Sei un Giudice Gastronomico. Valuta le dosi per {persone} persone e la sicurezza (vincoli: {vincoli}) delle ricette: {ricette}. Voto 1-10 e critica breve.")
                        judge_res = (judge_prompt | llm).invoke({
                            "persone": st.session_state.num_persone,
                            "vincoli": st.session_state.preferenze,
                            "ricette": chef_res.content
                        })
                        st.session_state.ultimo_giudizio = judge_res.content
                        st.session_state.total_tokens += judge_res.response_metadata.get('token_usage', {}).get('total_tokens', 0)

                st.session_state.chat_history.append({"role": "assistant", "content": chef_res.content})
                st.rerun()

            except Exception as e:
                if "429" in str(e):
                    st.error("⏳ Limite di velocità raggiunto (Rate Limit). Attendi 60 secondi prima di riprovare.")
                else:
                    st.error(f"⚠️ Errore di connessione o token: {e}")