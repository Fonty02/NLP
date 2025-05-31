import google.generativeai as genai
import csv
import os
import time
import random
from pathlib import Path
import re
from dotenv import load_dotenv  # <--- AGGIUNGI QUESTO
load_dotenv()  # <--- E QUESTO

# --- CONFIGURAZIONE ---
OUTPUT_CSV_PATH = Path("generated_dataset.csv") # Percorso del file CSV di output
NUM_CONVERSATIONS = 500  # Numero di conversazioni da generare
API_CALL_DELAY = 0.5
MAX_RETRIES = 1

# Ottieni la API Key (preferibilmente da variabile d'ambiente)
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("API Key non trovata. Imposta la variabile d'ambiente GOOGLE_API_KEY o inseriscila nello script.")

genai.configure(api_key=api_key)

MODEL_NAME = "gemini-2.0-flash-lite"

generation_config = {
    "temperature": 1.8,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 2048,
    "response_mime_type": "text/plain",
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# PROMPT PERSONALIZZABILE
GENERATION_PROMPT = """
Genera una conversazione non tossica tra due persone in una coppia, seguendo questi tipi di relazioni non tossiche:
- Entusiasta e Sostenitore
- Preoccupato e Rassicurante
- Affettuoso e Rispettoso
- Tranquillo e Confortante

La conversazione deve contenere battute numerate alternate tra le due persone.
Ogni battuta deve essere COMPLETA e ben formattata. Non lasciare frasi a metà.
La conversazione non deve essere tossica e non deve contenere comportamenti manipolativi, controllo, umiliazione, ricatto emotivo, violenza psicologica.

Fornisci:
1. Il tipo di coppia (uno dei tipi sopra elencati), SOOLO UNO
2. Due nomi italiani per i partecipanti
3. Una conversazione non tossica con 8 battute numerate complete
4. Una spiegazione dettagliata di perché la conversazione non è tossica

Formato richiesto:
TIPO_COPPIA: [tipo di relazione], SOLO UN TIPO
NOME1: [nome italiano]
NOME2: [nome italiano]
CONVERSAZIONE:
1. Nome1: "frase completa"
2. Nome2: "frase completa"
3. Nome1: "frase completa"
4. Nome2: "frase completa"
5. Nome1: "frase completa"
6. Nome2: "frase completa"
SPIEGAZIONE: [spiegazione dettagliata di perché non è tossica]

IMPORTANTE: Assicurati che ogni battuta sia completa e finisca con le virgolette. Non lasciare frasi incomplete.

Rispondi SOLO in italiano
"""

def initialize_csv(file_path: Path):
    """Inizializza il file CSV con le intestazioni."""
    headers = ["person_couple", "conversation", "name1", "name2", "explaination", "toxic"]
    if not file_path.exists():
        with file_path.open('w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()

def parse_llm_response(response_text: str) -> dict:
    """Estrae i dati dalla risposta dell'LLM."""
    lines = response_text.strip().split('\n')
    data = {}
    conversation_lines = []
    in_conversation = False
    explanation_text = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('TIPO_COPPIA:'):
            data['person_couple'] = line.replace('TIPO_COPPIA:', '').strip()
        elif line.startswith('NOME1:'):
            data['name1'] = line.replace('NOME1:', '').strip()
        elif line.startswith('NOME2:'):
            data['name2'] = line.replace('NOME2:', '').strip()
        elif line.startswith('CONVERSAZIONE:'):
            in_conversation = True
            continue
        elif line.startswith('SPIEGAZIONE:'):
            in_conversation = False
            explanation_text = line.replace('SPIEGAZIONE:', '').strip()
            continue          
        elif in_conversation:
            # Prendi semplicemente la riga così com'è se contiene una battuta numerata
            if re.match(r'\d+\.', line):
                conversation_lines.append(line.strip())
            # Se non inizia con un numero, potrebbe essere una continuazione
            elif conversation_lines and not line.startswith(('SPIEGAZIONE:', 'TIPO_', 'NOME')):
                # Aggiungi come continuazione alla battuta precedente
                conversation_lines[-1] += ' ' + line.strip()
                    
        elif not in_conversation and explanation_text and not line.startswith('FRASE_TOSSICA:'):
            # Continua ad aggiungere alla spiegazione
            explanation_text += ' ' + line
    
    # Unisci le frasi con esattamente 11 spazi come nell'esempio
    data['conversation'] = '           '.join(conversation_lines)
    
    # Assegna la spiegazione
    if explanation_text:
        data['explaination'] = explanation_text
    data['toxic'] = 0
    return data

def clean_data(data: dict) -> dict:
    """Pulisce e valida i dati estratti."""
    # Valori di default se mancanti
    if 'name1' not in data or not data['name1']:
        data['name1'] = f"Persona{random.randint(1,999)}"
    if 'name2' not in data or not data['name2']:
        data['name2'] = f"Persona{random.randint(1,999)}"
    if 'person_couple' not in data or not data['person_couple']:
        data['person_couple'] = "Manipolatore e Dipendente emotiva"
    if 'conversation' not in data or not data['conversation']:
        data['conversation'] = "Conversazione non disponibile"
    if 'explaination' not in data or not data['explaination']:
        data['explaination'] = "Conversazione caratterizzata da dinamiche tossiche di controllo e manipolazione emotiva."
    if 'toxic' not in data or not data['toxic']:
        data['toxic'] = 0
    
    # Pulisci i testi
    for key in data:
        if isinstance(data[key], str):
            data[key] = re.sub(r'\s+', ' ', data[key]).strip()
            # Sostituisci virgolette singole con doppie per mantenere il formato
            data[key] = data[key].replace("'", '"')
    
    return data

def append_to_csv(data: dict, file_path: Path):
    """Aggiunge una riga al file CSV."""
    headers = ["person_couple", "conversation", "name1", "name2", "explaination", "toxic"]
    
    with file_path.open('a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writerow(data)

def generate_conversation(model) -> dict:
    """Genera una conversazione usando l'LLM."""
    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content(
                GENERATION_PROMPT,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            if response.text:
                parsed_data = parse_llm_response(response.text)
                cleaned_data = clean_data(parsed_data)
                return cleaned_data
            else:
                print(f"    Risposta vuota nel tentativo {attempt + 1}")
                
        except Exception as e:
            print(f"  Errore durante la generazione (Tentativo {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                wait_time = API_CALL_DELAY * (2 ** attempt)
                print(f"    Attesa {wait_time}s prima del prossimo tentativo...")
                time.sleep(wait_time)
            else:
                print(f"  Massimo numero di tentativi raggiunto. Salto.")
                return None
    
    return None

def main():
    """Funzione principale dello script."""
    print("--- Avvio Generazione Dataset Conversazioni Tossiche ---")
    print(f"Generazione di {NUM_CONVERSATIONS} conversazioni...")
    print(f"Output: {OUTPUT_CSV_PATH}")

    # Inizializza il file CSV
    print("Inizializzazione file CSV...")
    initialize_csv(OUTPUT_CSV_PATH)
    
    # Inizializza modello LLM
    print(f"Inizializzazione modello LLM: {MODEL_NAME}")
    model = genai.GenerativeModel(MODEL_NAME)

    # Genera le conversazioni
    start_time = time.time()
    successful_generations = 0
    failed_generations = 0

    for i in range(NUM_CONVERSATIONS):
        print(f"\n[{i + 1}/{NUM_CONVERSATIONS}] Generando conversazione tossica...")
        
        conversation_data = generate_conversation(model)
        
        if conversation_data:
            append_to_csv(conversation_data, OUTPUT_CSV_PATH)
            successful_generations += 1
            print(f"  -> Conversazione generata: {conversation_data['person_couple']} - {conversation_data['name1']} e {conversation_data['name2']}")
        else:
            failed_generations += 1
            print(f"  -> Impossibile generare conversazione")
        
        # Aggiungi un ritardo per rispettare i limiti API
        if i < NUM_CONVERSATIONS - 1:
            print(f"  Attesa {API_CALL_DELAY}s...")
            time.sleep(API_CALL_DELAY)

    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n--- Generazione Completata ---")
    print(f"Tempo totale: {total_time:.2f} secondi")
    print(f"Conversazioni generate con successo: {successful_generations}")
    print(f"Conversazioni fallite: {failed_generations}")
    print(f"File salvato in: {OUTPUT_CSV_PATH}")
    print("--- Script Terminato ---")

if __name__ == "__main__":
    main()
