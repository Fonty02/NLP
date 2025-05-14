import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import os
from trl import SFTTrainer
from functools import partial # Per passare argomenti fissi a .map

# --- Configurazione Iniziale ---
MODEL_ID_GPTQ = "Qwen/Qwen2-1.5B"
MODEL_ID_FOR_QLORA_TRAINING = "Qwen/Qwen2-1.5B"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"VRAM available: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

# --- 1. Dataset di Esempio (CIPV) ---
CIPV_LABEL = "CIPV"
NON_CIPV_LABEL = "Non-CIPV"
POSSIBLE_LABELS = [CIPV_LABEL, NON_CIPV_LABEL]
# ... (definizione del dataset 'data' invariata) ...
data = {
    "train": [
        {"id": "train_01", "conversation": "Partner A: Non vali niente, sei inutile. Partner B: Smettila, mi fai stare male.", "label": CIPV_LABEL},
        {"id": "train_02", "conversation": "Partner A: Ti controllo il telefono stasera. Partner B: Non hai il diritto di farlo.", "label": CIPV_LABEL},
        {"id": "train_03", "conversation": "Partner A: Amore, ho preparato la cena! Partner B: Fantastico, grazie!", "label": NON_CIPV_LABEL},
        {"id": "train_04", "conversation": "Partner A: Sei sempre con i tuoi amici, non ti importa di me! Partner B: Non è vero, cerco solo di avere i miei spazi.", "label": CIPV_LABEL},
        {"id": "train_05", "conversation": "Partner A: Andiamo al cinema stasera? Partner B: Certo, ottima idea!", "label": NON_CIPV_LABEL},
        {"id": "train_06", "conversation": "Partner A: Se mi lasci, ti rovino. Partner B: Non puoi minacciarmi.", "label": CIPV_LABEL},
        {"id": "train_07", "conversation": "Partner A: Mi manchi tanto. Partner B: Anche tu, ci vediamo presto.", "label": NON_CIPV_LABEL},
        {"id": "train_08", "conversation": "Partner A: Non mi piace come ti vesti, sembri una poco di buono. Partner B: Io mi vesto come voglio.", "label": CIPV_LABEL},
        {"id": "train_09", "conversation": "Partner A: Se parli con lui un'altra volta, finisce male. Partner B: Non puoi dirmi con chi posso parlare.", "label": CIPV_LABEL},
        {"id": "train_10", "conversation": "Partner A: Hai fatto un ottimo lavoro con la presentazione! Partner B: Grazie per il supporto!", "label": NON_CIPV_LABEL},
        {"id": "train_11", "conversation": "Partner A: Non ti azzardare a mettere quella foto online o te ne pentirai. Partner B: Sono libera di pubblicare ciò che voglio.", "label": CIPV_LABEL},
        {"id": "train_12", "conversation": "Partner A: Ti ho preso un regalo a sorpresa! Partner B: Non vedo l'ora di scoprire cosa sia!", "label": NON_CIPV_LABEL},
        {"id": "train_13", "conversation": "Partner A: Perché ci hai messo così tanto a rispondere? Con chi stavi messaggiando? Partner B: Ero occupata, non devo giustificarmi sempre.", "label": CIPV_LABEL},
        {"id": "train_14", "conversation": "Partner A: Vuoi un tè mentre guardi il tuo programma preferito? Partner B: Sei un tesoro, grazie!", "label": NON_CIPV_LABEL},
        {"id": "train_15", "conversation": "Partner A: Sei grassa/o, nessuno ti guarderà mai oltre me. Partner B: Queste parole mi feriscono.", "label": CIPV_LABEL},
        {"id": "train_16", "conversation": "Partner A: Facciamo una passeggiata nel parco? Partner B: Volentieri, è una bella giornata!", "label": NON_CIPV_LABEL},
        {"id": "train_17", "conversation": "Partner A: Ti ho detto mille volte di non parlare quando ci sono i miei amici. Partner B: Ho il diritto di esprimere la mia opinione.", "label": CIPV_LABEL},
        {"id": "train_18", "conversation": "Partner A: Sei d'accordo se invitiamo i tuoi genitori domenica? Partner B: Ottima idea, saranno contenti!", "label": NON_CIPV_LABEL},
        {"id": "train_19", "conversation": "Partner A: Se non rispondi entro 5 minuti, vengo lì e faccio una scenata. Partner B: Non puoi controllarmi così.", "label": CIPV_LABEL},
        {"id": "train_20", "conversation": "Partner A: Ho prenotato per il nostro anniversario! Partner B: Che sorpresa meravigliosa!", "label": NON_CIPV_LABEL},
        {"id": "train_21", "conversation": "Partner A: Ti ho monitorato con un'app sul tuo telefono, so dove sei stata. Partner B: Questo è inquietante e invasivo.", "label": CIPV_LABEL},
        {"id": "train_22", "conversation": "Partner A: Rispetto la tua decisione di uscire con le tue amiche. Partner B: Grazie per capire che ho bisogno dei miei spazi.", "label": NON_CIPV_LABEL},
        {"id": "train_23", "conversation": "Partner A: Sei mia/o e di nessun altro, chiaro? Partner B: Non sono un oggetto di tua proprietà.", "label": CIPV_LABEL},
        {"id": "train_24", "conversation": "Partner A: Cosa ti piacerebbe fare questo weekend? Partner B: Potremmo fare un picnic al lago!", "label": NON_CIPV_LABEL},
        {"id": "train_25", "conversation": "Partner A: Cancella quell'amicizia sui social o ti lascio. Partner B: Non puoi dirmi chi posso avere come amico.", "label": CIPV_LABEL},
        {"id": "train_26", "conversation": "Partner A: Ti supporterò qualunque scelta tu faccia per la tua carriera. Partner B: Il tuo supporto significa molto per me.", "label": NON_CIPV_LABEL},
        {"id": "train_27", "conversation": "Partner A: Fammi vedere con chi stai messaggiando, subito! Partner B: Questi sono messaggi privati.", "label": CIPV_LABEL},
        {"id": "train_28", "conversation": "Partner A: Sai che ti amo anche quando non siamo d'accordo. Partner B: Anche io, è normale avere opinioni diverse.", "label": NON_CIPV_LABEL},
        {"id": "train_29", "conversation": "Partner A: Se mi lasci, diffonderò le tue foto intime. Partner B: Questo è ricatto, non puoi farlo.", "label": CIPV_LABEL},
        {"id": "train_30", "conversation": "Partner A: Mi piace quando condividiamo le nostre giornate. Partner B: È bello potersi confrontare con te.", "label": NON_CIPV_LABEL},
        {"id": "train_31", "conversation": "Partner A: Ti proibisco di andare a quella festa. Partner B: Non puoi decidere per me.", "label": CIPV_LABEL},
        {"id": "train_32", "conversation": "Partner A: Cuciniamo insieme stasera? Partner B: Sì, sarà divertente!", "label": NON_CIPV_LABEL},
        {"id": "train_33", "conversation": "Partner A: Sei patetica/o quando piangi, sembri una bambina/o. Partner B: Non dovresti sminuire i miei sentimenti.", "label": CIPV_LABEL},
        {"id": "train_34", "conversation": "Partner A: Possiamo parlare di ciò che è successo? Partner B: Certo, è importante comunicare.", "label": NON_CIPV_LABEL},
        {"id": "train_35", "conversation": "Partner A: Non ti permetterò mai di lasciarmi, piuttosto ti faccio del male. Partner B: Questo è terrificante, mi stai minacciando.", "label": CIPV_LABEL},
        {"id": "train_36", "conversation": "Partner A: Ti ho preso questi fiori perché mi ricordavano il tuo sorriso. Partner B: Che pensiero dolce, grazie!", "label": NON_CIPV_LABEL},
        {"id": "train_37", "conversation": "Partner A: Dovresti essere grata/o che sto con te, nessun altro ti vorrebbe. Partner B: Questo non è vero e mi ferisce.", "label": CIPV_LABEL},
        {"id": "train_38", "conversation": "Partner A: Rispetterò sempre i tuoi confini personali. Partner B: Apprezzo molto questo tuo atteggiamento.", "label": NON_CIPV_LABEL},
        {"id": "train_39", "conversation": "Partner A: Vedo che ti piace civettare con tutti, sei solo una poco di buono. Partner B: Non ti permetto di insultarmi così.", "label": CIPV_LABEL},
        {"id": "train_40", "conversation": "Partner A: Mi piace quando discutiamo delle nostre differenze con rispetto. Partner B: È uno degli aspetti più belli del nostro rapporto.", "label": NON_CIPV_LABEL},
        {"id": "train_41", "conversation": "Partner A: Devi chiedere il mio permesso prima di uscire. Partner B: Sono adulta/o e posso decidere da sola/o.", "label": CIPV_LABEL},
        {"id": "train_42", "conversation": "Partner A: Ti va di provare quel nuovo ristorante? Partner B: Sì, ho sentito che è ottimo!", "label": NON_CIPV_LABEL},
        {"id": "train_43", "conversation": "Partner A: Se mi tradisci, giuro che ti ammazzo. Partner B: Mi stai spaventando con queste minacce.", "label": CIPV_LABEL},
        {"id": "train_44", "conversation": "Partner A: Sono fiero/a dei progressi che stai facendo nel tuo lavoro! Partner B: Grazie per il tuo incoraggiamento costante.", "label": NON_CIPV_LABEL},
        {"id": "train_45", "conversation": "Partner A: Sei stupida/o, non capisci mai niente. Partner B: Non merito di essere trattata/o così.", "label": CIPV_LABEL},
        {"id": "train_46", "conversation": "Partner A: Ti ascolto e cerco di capire il tuo punto di vista. Partner B: Apprezzo molto la tua apertura mentale.", "label": NON_CIPV_LABEL},
        {"id": "train_47", "conversation": "Partner A: Ho installato delle telecamere in casa per tenerti d'occhio. Partner B: Questo è assurdo e inquietante.", "label": CIPV_LABEL},
        {"id": "train_48", "conversation": "Partner A: Ti aiuto con le faccende così possiamo avere più tempo insieme dopo. Partner B: Grazie, sei sempre così premuroso/a.", "label": NON_CIPV_LABEL},
        {"id": "train_49", "conversation": "Partner A: Non meriti di essere amata/o, sei un fallimento. Partner B: Le tue parole mi feriscono profondamente.", "label": CIPV_LABEL},
        {"id": "train_50", "conversation": "Partner A: Sono qui per te se hai bisogno di parlare. Partner B: È bello sapere di poterti contare.", "label": NON_CIPV_LABEL},
    ],
    "test": [
        {"id": "test_01", "conversation": "Partner A: Ti spacco la faccia se non fai come dico io. Partner B: Lasciami in pace!", "label": CIPV_LABEL},
        {"id": "test_02", "conversation": "Partner A: Stasera pizza e film? Partner B: Perfetto!", "label": NON_CIPV_LABEL},
        {"id": "test_03", "conversation": "Partner A: Devi dirmi tutte le tue password, altrimenti chissà cosa combini. Partner B: È una violazione della mia privacy.", "label": CIPV_LABEL},
        {"id": "test_04", "conversation": "Partner A: Ti amo da impazzire! Partner B: Anch'io tesoro!", "label": NON_CIPV_LABEL},
        {"id": "test_05", "conversation": "Partner A: Se mi lasci mi uccido, sarà colpa tua. Partner B: Non puoi farmi questo ricatto emotivo.", "label": CIPV_LABEL},
        {"id": "test_06", "conversation": "Partner A: Posso aiutarti con il trasloco questo weekend? Partner B: Sarebbe fantastico, grazie!", "label": NON_CIPV_LABEL},
        {"id": "test_07", "conversation": "Partner A: Chi era quel tizio con cui parlavi? Ti tengo d'occhio. Partner B: Era solo un collega, non devi essere così geloso.", "label": CIPV_LABEL},
        {"id": "test_08", "conversation": "Partner A: Ho fatto la spesa e ho preso i tuoi biscotti preferiti. Partner B: Sei sempre così attento/a, grazie!", "label": NON_CIPV_LABEL},
        {"id": "test_09", "conversation": "Partner A: Non ti permetterò di andare a quella riunione di lavoro, ci sono troppi uomini. Partner B: Non puoi impedirmi di fare il mio lavoro.", "label": CIPV_LABEL},
        {"id": "test_10", "conversation": "Partner A: Mi piace come affrontiamo insieme i problemi. Partner B: Sì, siamo una bella squadra.", "label": NON_CIPV_LABEL},
        {"id": "test_11", "conversation": "Partner A: Hai indossato quella gonna apposta per attirare l'attenzione, vero? Sei disgustosa. Partner B: Indosso ciò che mi fa sentire bene.", "label": CIPV_LABEL},
        {"id": "test_12", "conversation": "Partner A: Buongiorno! Ti ho preparato il caffè come piace a te. Partner B: Che dolce pensiero per iniziare la giornata!", "label": NON_CIPV_LABEL},
        {"id": "test_13", "conversation": "Partner A: Ti ho seguito fino all'ufficio oggi, volevo vedere cosa fai realmente. Partner B: Questo è inquietante e mi fa paura.", "label": CIPV_LABEL},
        {"id": "test_14", "conversation": "Partner A: Vuoi parlare di come ti senti riguardo a quella situazione? Partner B: Sì, mi aiuterebbe confrontarmi con te.", "label": NON_CIPV_LABEL},
        {"id": "test_15", "conversation": "Partner A: Se parli ancora con la tua ex, ti lascio immediatamente. Partner B: Non puoi controllare chi frequento.", "label": CIPV_LABEL},
    ]
}
train_dataset = Dataset.from_list(data["train"])
test_dataset = Dataset.from_list(data["test"])
dataset = DatasetDict({"train": train_dataset, "test": test_dataset})


# --- 2. Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID_GPTQ, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    # Per Qwen, è importante anche che pad_token_id sia impostato correttamente
    # Se si usa eos_token come pad_token, anche pad_token_id deve essere eos_token_id
    tokenizer.pad_token_id = tokenizer.eos_token_id


# --- 3. Funzione di Prompting e Generazione per la Classificazione (invariata) ---
def create_prompt_for_classification(conversation_text):
    messages = [
        {"role": "system", "content": f"Sei un assistente che classifica conversazioni. Rispondi solo con '{CIPV_LABEL}' o '{NON_CIPV_LABEL}'."},
        {"role": "user", "content": f"Analizza la seguente conversazione e determina se contiene linguaggio tossico o elementi di Cyber Intimate Partner Violence (CIPV).\n\nConversazione:\n{conversation_text}\n\nClassificazione:"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def get_prediction(model, tokenizer, conversation_text):
    prompt = create_prompt_for_classification(conversation_text)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id, # Assicurati che eos_token_id sia corretto per Qwen
            do_sample=False
        )
    response_ids = outputs[0, inputs.input_ids.shape[1]:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    if CIPV_LABEL.lower() in response_text.lower():
        return CIPV_LABEL
    elif NON_CIPV_LABEL.lower() in response_text.lower():
        return NON_CIPV_LABEL
    else:
        return NON_CIPV_LABEL

# --- 4. Valutazione del Modello Pre-addestrato (Baseline) (invariata) ---
print("\n--- Valutazione Modello Pre-addestrato (Baseline) ---")
# ... (codice di valutazione baseline invariato) ...
base_model_for_eval = AutoModelForCausalLM.from_pretrained(
    MODEL_ID_GPTQ,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
    trust_remote_code=True,
    attn_implementation="eager" 
)


base_model_for_eval.eval()

y_true_base, y_pred_base = [], []
for example in dataset["test"]:
    conversation, true_label = example["conversation"], example["label"]
    predicted_label = get_prediction(base_model_for_eval, tokenizer, conversation)
    y_true_base.append(true_label)
    y_pred_base.append(predicted_label)

accuracy_base = accuracy_score(y_true_base, y_pred_base)
precision_base, recall_base, f1_base, _ = precision_recall_fscore_support(
    y_true_base, y_pred_base, average='weighted', labels=POSSIBLE_LABELS, zero_division=0
)
print(f"Baseline ({MODEL_ID_GPTQ}) - Accuracy: {accuracy_base:.4f}, Precision: {precision_base:.4f}, Recall: {recall_base:.4f}, F1-score: {f1_base:.4f}")

del base_model_for_eval
if DEVICE == "cuda": torch.cuda.empty_cache()


# --- 5. Fine-tuning con QLoRA ---
print("\n--- Fine-tuning con QLoRA ---")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
    bnb_4bit_use_double_quant=True,
)

print(f"Caricamento modello base {MODEL_ID_FOR_QLORA_TRAINING} per QLoRA...")
model_for_qlora = AutoModelForCausalLM.from_pretrained(
    MODEL_ID_FOR_QLORA_TRAINING,
    quantization_config=bnb_config,
    device_map={"":0},
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
    trust_remote_code=True,
    attn_implementation="eager" 
)
model_for_qlora.config.use_cache = False
model_for_qlora.config.pretraining_tp = 1

model_for_qlora = prepare_model_for_kbit_training(model_for_qlora, use_gradient_checkpointing=True)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
peft_model = get_peft_model(model_for_qlora, lora_config)
peft_model.print_trainable_parameters()

# --- NUOVA SEZIONE: Preparazione Dataset Pre-Tokenizzato ---
MAX_SEQ_LEN_FOR_TRAINING = 512 # Definisci la lunghezza massima per il training

def format_and_tokenize_entry(example, tokenizer, max_length):
    messages = [
        {"role": "system", "content": f"Sei un assistente che classifica conversazioni. Rispondi solo con '{CIPV_LABEL}' o '{NON_CIPV_LABEL}'."},
        {"role": "user", "content": f"Analizza la seguente conversazione e determina se contiene linguaggio tossico o elementi di Cyber Intimate Partner Violence (CIPV).\n\nConversazione:\n{example['conversation']}\n\nClassificazione:"},
        {"role": "assistant", "content": example['label']}
    ]
    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
    tokenized_inputs = tokenizer(
        formatted_text,
        truncation=True,
        padding=False, # Non paddare qui, DataCollator lo farà per batch
        max_length=max_length,
    )
    return tokenized_inputs

# Crea una funzione parziale con tokenizer e max_length già impostati
tokenize_function_with_args = partial(format_and_tokenize_entry, tokenizer=tokenizer, max_length=MAX_SEQ_LEN_FOR_TRAINING)

print("Formattazione e tokenizzazione del dataset di training...")
formatted_train_dataset = dataset["train"].map(
    tokenize_function_with_args,
    batched=False, # Puoi mettere True se la tua funzione è ottimizzata per batch e restituisce una lista di dicts
    remove_columns=dataset["train"].column_names # Rimuove le vecchie colonne
)
print(f"Dataset di training formattato. Esempio: {formatted_train_dataset[0]}")
# --- FINE NUOVA SEZIONE ---


training_args = TrainingArguments(
    output_dir="./qwen_cipv_finetuned_qlora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=50,
    fp16=not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    gradient_checkpointing=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    report_to="none",
)

print("Inizio costruzione SFTTrainer...")
trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=formatted_train_dataset, # Passa il dataset pre-tokenizzato
    # eval_dataset=formatted_eval_dataset, # Se vuoi valutare, pre-tokenizza anche questo
    processing_class=tokenizer, # Usa l'argomento raccomandato al posto di 'tokenizer'
    peft_config=lora_config,
    # dataset_text_field="text", # Rimosso, il dataset è tokenizzato
    # max_seq_length=MAX_SEQ_LEN_FOR_TRAINING, # Rimosso, gestito in pre-tokenizzazione
)
print("SFTTrainer costruito.")

print("Inizio fine-tuning...")
trainer.train()
print("Fine-tuning completato.")

# --- 6. Salvare il modello (solo gli adapter LoRA) (invariato) ---
FINETUNED_ADAPTERS_PATH = "./qwen_cipv_qlora_adapters"
trainer.save_model(FINETUNED_ADAPTERS_PATH)
print(f"Adapter LoRA salvati in {FINETUNED_ADAPTERS_PATH}")

del model_for_qlora, peft_model, trainer
if DEVICE == "cuda": torch.cuda.empty_cache()

# --- 7. Valutazione del Modello Fine-tuned (invariata) ---
print("\n--- Valutazione Modello Fine-tuned con QLoRA ---")
# ... (codice di valutazione del modello fine-tuned invariato) ...
base_model_for_tuned_eval = AutoModelForCausalLM.from_pretrained(
    MODEL_ID_GPTQ, # Usiamo il modello GPTQ-Int8 come base
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
    trust_remote_code=True,
    attn_implementation="eager" 
)
from peft import PeftModel
tuned_model = PeftModel.from_pretrained(base_model_for_tuned_eval, FINETUNED_ADAPTERS_PATH)
tuned_model.eval()

y_true_tuned, y_pred_tuned = [], []
for example in dataset["test"]:
    conversation, true_label = example["conversation"], example["label"]
    predicted_label = get_prediction(tuned_model, tokenizer, conversation)
    y_true_tuned.append(true_label)
    y_pred_tuned.append(predicted_label)

accuracy_tuned = accuracy_score(y_true_tuned, y_pred_tuned)
precision_tuned, recall_tuned, f1_tuned, _ = precision_recall_fscore_support(
    y_true_tuned, y_pred_tuned, average='weighted', labels=POSSIBLE_LABELS, zero_division=0
)
print(f"Fine-tuned ({MODEL_ID_GPTQ} + LoRA) - Accuracy: {accuracy_tuned:.4f}, Precision: {precision_tuned:.4f}, Recall: {recall_tuned:.4f}, F1-score: {f1_tuned:.4f}")


print("\n--- Confronto Performance ---")
print(f"Baseline Model ({MODEL_ID_GPTQ}):")
print(f"  Accuracy: {accuracy_base:.4f}, Precision: {precision_base:.4f}, Recall: {recall_base:.4f}, F1: {f1_base:.4f}")
print(f"Fine-tuned Model (QLoRA trained on {MODEL_ID_FOR_QLORA_TRAINING}, adapters on {MODEL_ID_GPTQ}):")
print(f"  Accuracy: {accuracy_tuned:.4f}, Precision: {precision_tuned:.4f}, Recall: {recall_tuned:.4f}, F1: {f1_tuned:.4f}")

print("\nTask completato.")