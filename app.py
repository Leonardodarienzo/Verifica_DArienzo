import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import json

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="AI Kitchen Multi-Agent Pro", layout="wide", page_icon="👨‍🍳")

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
    st.session_state.ultimo_giudizio = "In attesa di una proposta dello Chef..."

# --- AGENTE 1: ESTRAZIONE DATI (Entity Extraction) ---
def update_kitchen_state(text, api_key):
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=api_key)
    extract_prompt = ChatPromptTemplate.from_template("""
    Analizza il messaggio dell'utente per aggiornare lo stato della cucina.
    REGOLE: 
    - Converti numeri in lettere in cifre (es: due -> 2, mezzo chilo -> 0.5 kg).
    - Estrai ingredienti, quantita', scadenze, persone e vincoli (celiaco, vegano).
    
    Rispondi SOLO con JSON:
    {{
        "ingredients": [{{ "item": "nome", "qty": "quantita", "expiry": "scadenza o null" }}],
        "preferences": ["stringa"],
        "people": "numero o null"
    }}
    Messaggio: {input}
    """)
    chain = extract_prompt | llm
    try:
        response = chain.invoke({"input": text})
        json_clean = response.content.strip().replace('```json', '').replace('```', '')
        data = json.loads(json_clean)
        
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
    except: pass

# --- SIDEBAR (SINISTRA): STATO DISPENSA ---
with st.sidebar:
    st.header("🛒 Inventario & Vincoli")
    st.info(f"👥 **Commensali:** {st.session_state.num_persone}")
    
    st.subheader("📦 Dispensa")
    if not st.session_state.dispensa:
        st.write("In attesa di dati...")
    else:
        for ing in st.session_state.dispensa:
            scad = ing['expiry'] if (ing['expiry'] and ing['expiry'] != 'null') else "N/D"
            st.success(f"**{ing['item']}**\n{ing['qty']} | Scad: {scad}")
    
    st.subheader("🚫 Vincoli Dietetici")
    for pref in st.session_state.preferenze:
        st.error(pref)
    
    st.divider()
    api_key = st.text_input("Groq API Key", type="password")
    if st.button("🔄 Reset Sessione"):
        st.session_state.clear()
        st.rerun()

# --- LAYOUT PRINCIPALE: CHAT (CENTRO) E GIUDICE (DESTRA) ---
col_chat, col_judge = st.columns([0.7, 0.3])

with col_chat:
    st.title("👨‍🍳 AI Chef Advisor")
    st.markdown("Analisi bilanciata delle porzioni e dei vincoli dietetici.")

    # Visualizzazione Chat
    for message in st.session_state.chat_history:
        avatar = "👨‍🍳" if message["role"] == "assistant" else None
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    user_input = st.chat_input("Inserisci dati o chiedi ricette...")

with col_judge:
    st.header("⚖️ Verifica Giudice")
    st.markdown("---")
    st.warning(st.session_state.ultimo_giudizio)

# --- LOGICA INTERAZIONE ---
if user_input:
    with col_chat:
        if not api_key:
            st.error("Inserisci l'API Key nella sidebar!")
        else:
            update_kitchen_state(user_input, api_key)
            st.chat_message("user").markdown(user_input)
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            try:
                llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=api_key)
                
                # Controllo soglia (Necessari dati minimi per procedere)
                sufficiente = "SI" if (len(st.session_state.dispensa) >= 5 and st.session_state.num_persone != "Non specificato") else "NO"
                
                # --- AGENTE 2: CHEF (Generazione con focus sulle porzioni) ---
                with st.spinner("Lo Chef sta calcolando le dosi..."):
                    disp_txt = "\n".join([f"- {i['item']} ({i['qty']}, scad: {i['expiry']})" for i in st.session_state.dispensa])
                    chef_prompt = ChatPromptTemplate.from_messages([
                        ("system", f"""Sei uno Chef esperto in nutrizione e gestione delle risorse.
                        DATI: Dispensa: {disp_txt} | Vincoli: {st.session_state.preferenze} | Persone: {st.session_state.num_persone}.
                        
                        REGOLE FONDAMENTALI:
                        1. Se SOGLIA={sufficiente} è NO: Chiedi dati mancanti (persone, scadenze, ecc.).
                        2. Se SOGLIA=SI: Proponi ESATTAMENTE 3 RICETTE COMPLETE.
                           - IMPORTANTE: Calcola DOSI REALISTICHE per {st.session_state.num_persone} persone. Non esagerare (es. non usare 1kg di verdure se ne bastano 400g).
                           - Specifica le quantita' esatte da usare per ogni ingrediente nella ricetta.
                           - Fornisci uno SVOLGIMENTO DETTAGLIATO per ogni piatto.
                        3. Rispetta rigorosamente Celiaci (No Glutine) e Vegani (No Animali).
                        4. Dai priorita' assoluta agli ingredienti in scadenza imminente.
                        5. Rispondi in italiano."""),
                        MessagesPlaceholder(variable_name="history"),
                        ("human", "{input}")
                    ])
                    history_lc = [HumanMessage(content=m["content"]) if m["role"]=="user" else AIMessage(content=m["content"]) for m in st.session_state.chat_history[:-1]]
                    chef_res = (chef_prompt | llm).invoke({"input": user_input, "history": history_lc})

                # --- AGENTE 3: GIUDICE (Riflessione critica sulle porzioni) ---
                if sufficiente == "SI":
                    with st.spinner("Il Giudice sta verificando le porzioni..."):
                        judge_prompt = ChatPromptTemplate.from_template("""
                        Sei un Giudice Gastronomico rigoroso. Valuta le 3 ricette dello Chef.
                        Persone: {persone}, Vincoli: {vincoli}.
                        
                        RICETTE DA ANALIZZARE:
                        {ricette}
                        
                        CRITERI DI VALUTAZIONE:
                        1. DOSI: Le quantita' suggerite sono corrette per {persone} persone o sono eccessive/insufficienti?
                        2. SICUREZZA: Sono stati usati ingredienti vietati (Glutine/Animali)?
                        3. SCADENZE: Lo Chef ha dato priorita' ai prodotti vicini alla scadenza?
                        
                        Fornisci un voto da 1 a 10 e un giudizio tecnico sintetico focalizzato sull'equilibrio delle porzioni.
                        """)
                        judge_res = (judge_prompt | llm).invoke({
                            "persone": st.session_state.num_persone,
                            "vincoli": st.session_state.preferenze,
                            "ricette": chef_res.content
                        })
                        st.session_state.ultimo_giudizio = judge_res.content

                # Salvataggio e aggiornamento UI
                st.session_state.chat_history.append({"role": "assistant", "content": chef_res.content})
                st.rerun()

            except Exception as e:
                st.error(f"Errore tecnico: {e}")