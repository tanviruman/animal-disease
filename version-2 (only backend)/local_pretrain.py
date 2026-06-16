import os
import re
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

# Set paths based on your workspace setup
DATA_DIR = "./DataSet of Animals" 
OUTPUT_DIR = "./assets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# MASTER SYMPTOM VOCABULARY
# Maps raw/variant text found in CSVs -> canonical symptom name used in model
# ══════════════════════════════════════════════════════════════════════════════
SYMPTOM_MAP = {
    # General & Respiratory
    'coughing': 'Coughing',
    'sneezing': 'Sneezing',
    'labored breathing': 'Labored Breathing',
    'nasal discharge': 'Nasal Discharge',
    'eye discharge': 'Eye Discharge',
    
    # Digestive
    'vomiting': 'Vomiting',
    'diarrhea': 'Diarrhea',
    'appetite loss': 'Appetite Loss',
    'loss of appetite': 'Appetite Loss',
    'reduced appetite': 'Appetite Loss',
    'increased appetite': 'Increased Appetite',
    'dehydration': 'Dehydration',
    'digestive issues': 'Digestive Issues',
    'excessive drooling': 'Excessive Drooling',
    
    # Systemic & Behavioral
    'fever': 'Fever',
    'lethargy': 'Lethargy',
    'weight loss': 'Weight Loss',
    'swelling': 'Swelling',
    'swollen joints': 'Swollen Joints',
    'swollen legs': 'Swollen Legs',
    'skin lesions': 'Skin Lesions',
    'skin irritations': 'Skin Lesions',
    'lameness': 'Lameness',
    'reduced mobility': 'Lameness',
    'mobility problems': 'Lameness',
    'limping': 'Lameness',
    'itching': 'Itching / Scratching',
    'scratching': 'Itching / Scratching',
    'hair loss': 'Hair Loss',
    'weakness': 'Weakness / Stiffness',
    'stiffness': 'Weakness / Stiffness',

    # Livestock specifics
    'parasites': 'Parasites',
    'ear infections': 'Ear Infections',
    'decreased milk yield': 'Decreased Milk Yield',
    'reduced milk production': 'Decreased Milk Yield',
    'reduced wool growth': 'Reduced Wool Production',
    'reduced wool production': 'Reduced Wool Production',
    
    # Pregnancy specifics
    'nesting': 'Nesting Behavior',
    'nesting behavior': 'Nesting Behavior',
    'restless': 'Restless Behavior',
    'restless behavior': 'Restless Behavior',
    'aggressive': 'Aggressive Behavior',
    'aggressive behavior': 'Aggressive Behavior',
    'clear': 'Clear Vaginal Discharge',
    'clear vaginal discharge': 'Clear Vaginal Discharge',
    'bloody': 'Bloody Vaginal Discharge',
    'bloody vaginal discharge': 'Bloody Vaginal Discharge',
    'purulent vaginal discharge': 'Purulent Vaginal Discharge',
    'purulent': 'Purulent Vaginal Discharge',
    'detected': 'Fetal Heart Sound Detected',
    'fetal heart sound detected': 'Fetal Heart Sound Detected'
}

# ══════════════════════════════════════════════════════════════════════════════
# SYNONYM RULES — Used by the API for fast fuzzy text matching
# Each symptom maps to a list of natural language expressions that imply it.
# This is exported to encoders.json to keep backend & pretrain in sync.
# ══════════════════════════════════════════════════════════════════════════════
SYNONYM_RULES = {
    "Aggressive Behavior": [
        "aggressive", "aggression", "growl", "bite", "biting", "snap", "snapping", "vicious", "hostile", "lunging"
    ],
    "Appetite Loss": [
        "appetite loss", "loss of appetite", "not eating", "refuse food", "decreased appetite",
        "reduced appetite", "off food", "stop eating", "wont eat", "won't eat", "refusing food", "off their food"
    ],
    "Bloody Vaginal Discharge": [
        "bloody discharge", "blood discharge", "vaginal bleeding", "bleeding from vagina", "bloody vaginal"
    ],
    "Clear Vaginal Discharge": [
        "clear discharge", "mucus discharge", "vaginal discharge", "clear vaginal"
    ],
    "Purulent Vaginal Discharge": [
        "purulent discharge", "pus discharge", "green discharge", "yellow discharge", "smelly discharge",
        "vaginal pus", "infected discharge"
    ],
    "Coughing": [
        "coughing", "cough", "coughed", "hacking", "kennel cough", "honking cough"
    ],
    "Decreased Milk Yield": [
        "decreased milk", "less milk", "low milk", "milk drop", "milk production", "dry up milk"
    ],
    "Dehydration": [
        "dehydration", "dehydrated", "dry gums", "sunken eyes", "thirsty", "not drinking", "tacky gums"
    ],
    "Diarrhea": [
        "diarrhea", "diarrhoea", "loose stool", "watery poop", "runny stool", "runny poop", "scours",
        "soft stool", "liquid stool", "bloody stool"
    ],
    "Digestive Issues": [
        "digestive issues", "stomach ache", "indigestion", "bloated", "gas", "digestive", "gurgling stomach",
        "abdominal pain", "stomach pain", "tummy ache", "belching", "burping"
    ],
    "Ear Infections": [
        "ear infection", "ear infections", "scratching ear", "shaking head", "smelly ears",
        "ear discharge", "ear odor", "dark ear wax", "head shaking"
    ],
    "Excessive Drooling": [
        "drooling", "drool", "excessive drooling", "hypersalivation", "salivating", "saliva", "slobbering"
    ],
    "Eye Discharge": [
        "eye discharge", "runny eyes", "watery eyes", "goopy eyes", "eye gunk", "tearing",
        "crusty eyes", "eye mucus", "eye boogers"
    ],
    "Fetal Heart Sound Detected": [
        "fetal heart", "puppy heartbeat", "heartbeat detected", "heart sound", "fetal movement"
    ],
    "Fever": [
        "fever", "hot", "warm to touch", "high temperature", "feverish", "high temp", "feels warm", "burning up"
    ],
    "Hair Loss": [
        "hair loss", "losing fur", "bald spots", "bald patch", "alopecia", "missing hair",
        "fur loss", "patchy coat", "coat loss", "thinning coat"
    ],
    "Increased Appetite": [
        "increased appetite", "always hungry", "hungry", "eating more", "voracious", "ravenous", "polyphagia"
    ],
    "Itching / Scratching": [
        "itching", "itchy", "scratching", "itch", "scratches", "rubbing skin", "rubbing face",
        "scratches all the time", "constant scratching", "biting skin", "chewing paws"
    ],
    "Labored Breathing": [
        "labored breathing", "breathing hard", "pant", "panting", "short of breath", "difficulty breathing",
        "wheezing", "heavy breathing", "dyspnea", "gasping", "rapid breathing", "open mouth breathing"
    ],
    "Lameness": [
        "lameness", "limping", "lame", "favoring leg", "difficulty walking", "trouble walking",
        "cannot walk", "reduced mobility", "stiff legs", "not bearing weight"
    ],
    "Lethargy": [
        "lethargy", "lethargic", "sluggish", "tired", "no energy", "weakness", "inactive",
        "passive", "sleepy", "depressed", "dull", "unresponsive", "listless"
    ],
    "Nasal Discharge": [
        "nasal discharge", "runny nose", "snot", "stuffy nose", "discharge from nose",
        "nasal mucus", "congested", "nose dripping"
    ],
    "Nesting Behavior": [
        "nesting", "nesting behavior", "nest", "building nest", "gathering bedding", "making a nest"
    ],
    "Parasites": [
        "parasites", "worms", "fleas", "ticks", "lice", "mite", "mites", "parasite",
        "intestinal worms", "roundworms", "tapeworms", "heartworm"
    ],
    "Reduced Wool Production": [
        "wool drop", "less wool", "reduced wool", "poor fleece", "thin fleece"
    ],
    "Restless Behavior": [
        "restless", "pacing", "cannot settle", "can't settle", "restlessness", "agitated",
        "pacing back and forth", "cannot rest"
    ],
    "Skin Lesions": [
        "skin lesions", "skin irritations", "rash", "scabs", "lesion", "lesions",
        "skin sores", "wounds on skin", "crusty skin", "skin bumps"
    ],
    "Sneezing": [
        "sneezing", "sneeze", "sneezed", "reverse sneeze"
    ],
    "Swelling": [
        "swelling", "swollen", "lump", "bump", "swelled", "mass", "growth", "abscess"
    ],
    "Swollen Joints": [
        "swollen joints", "joint pain", "stiff joints", "joint swelling", "arthritis pain"
    ],
    "Swollen Legs": [
        "swollen legs", "leg swelling", "puffy legs", "edema in legs"
    ],
    "Vomiting": [
        "vomiting", "vomit", "throwing up", "threw up", "puking", "puke", "vomited",
        "throwingup", "retching", "regurgitating", "regurgitation", "bring up food"
    ],
    "Weakness / Stiffness": [
        "weakness", "weak", "stiffness", "stiff", "wobbly", "unstable on feet", "muscle weakness",
        "trembling", "shaking", "shivering", "unable to stand", "collapsing"
    ],
    "Weight Loss": [
        "weight loss", "losing weight", "skinny", "thin", "lost weight", "wasting", "emaciated"
    ],
}


def clean_disease_name(name):
    if pd.isna(name):
        return "Healthy / No Disease"
    s = str(name).replace('-', ' ').replace('_', ' ').strip().lower()
    s = re.sub(r'\s+', ' ', s)
    words = s.split()
    cap_words = []
    for i, w in enumerate(words):
        if i == 0 or i == len(words) - 1:
            cap_words.append(w.capitalize())
        elif w in {'and', 'or', 'of', 'in', 'to', 'for', 'with', 'on', 'at', 'by', 'from', 'a', 'an', 'the'}:
            cap_words.append(w)
        else:
            cap_words.append(w.capitalize())
    res = " ".join(cap_words)
    # Disease name normalization aliases
    if res in ["Bluetongue Virus", "Blue Tongue Virus", "Blue Tongue Disease", "Blue Tongue"]:
        return "Bluetongue"
    if res == "Canine Infectious Hepatitis":
        return "Canine Hepatitis"
    if res in ["Caprine Arthritis Encephalitis Virus", "Caprine Viral Arthritis", "Caprine Arthritis"]:
        return "Caprine Arthritis Encephalitis"
    if res == "Feline Chlamydiosis":
        return "Feline Chlamydia"
    if res in ["Rabbit Viral Hemorrhagic Disease", "Viral Hemorrhagic Disease"]:
        return "Rabbit Hemorrhagic Disease"
    if res == "Scrapie Disease":
        return "Scrapie"
    if res == "Porcine Epidemic Diarrhea Virus":
        return "Porcine Epidemic Diarrhea"
    return res

HEALTHY_VITALS = {
    'Dog': {'temp': 38.5, 'hr': 85}, 'Cat': {'temp': 38.6, 'hr': 120}, 'Cow': {'temp': 38.5, 'hr': 60},
    'Horse': {'temp': 38.0, 'hr': 36}, 'Rabbit': {'temp': 39.2, 'hr': 200}, 'Sheep': {'temp': 39.0, 'hr': 75},
    'Goat': {'temp': 39.1, 'hr': 80}, 'Pig': {'temp': 39.2, 'hr': 70}, 'Ferret': {'temp': 38.8, 'hr': 220}
}

def clean_temp(val):
    if pd.isna(val): return None
    val_str = str(val).upper()
    match = re.search(r'([0-9\.]+)', val_str)
    if not match: return None
    num = float(match.group(1))
    if 'F' in val_str or num > 50:
        num = (num - 32) * 5 / 9
    return round(num, 2)


def extract_symptoms_from_text(text_lower):
    """Extract a list of canonical symptoms from raw text using SYNONYM_RULES."""
    found = []
    for sym, keywords in SYNONYM_RULES.items():
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                if sym not in found:
                    found.append(sym)
                break
    return found


def build_and_train():
    records = []
    text_corpus = []
    text_labels = []  # List of List[str] (specific symptoms per document)

    # ── 1. PRIMARY CLEANED DISEASE PREDICTION CSV ────────────────────────────
    p1 = os.path.join(DATA_DIR, "cleaned_animal_disease_prediction.csv")
    if os.path.exists(p1):
        df = pd.read_csv(p1)
        for _, row in df.iterrows():
            species = str(row.get('Animal_Type', 'Dog')).strip()
            disease = clean_disease_name(row.get('Disease_Prediction'))
            temp = clean_temp(row.get('Body_Temperature')) or HEALTHY_VITALS.get(species, {'temp':38.5})['temp']
            hr = pd.to_numeric(row.get('Heart_Rate'), errors='coerce') or HEALTHY_VITALS.get(species, {'hr':85})['hr']
            
            active = []
            for col in ['Appetite_Loss','Vomiting','Diarrhea','Coughing','Labored_Breathing','Lameness','Skin_Lesions','Nasal_Discharge','Eye_Discharge']:
                clean_col = col.replace('_', ' ').lower()
                if str(row.get(col, '')).strip().lower() == 'yes' and clean_col in SYMPTOM_MAP:
                    active.append(SYMPTOM_MAP[clean_col])
            for s_col in ['Symptom_1','Symptom_2','Symptom_3','Symptom_4']:
                s_val = str(row.get(s_col, '')).strip().lower()
                if s_val in SYMPTOM_MAP:
                    active.append(SYMPTOM_MAP[s_val])
            records.append({'species': species, 'disease': disease, 'temp': float(temp), 'hr': float(hr), 'symptoms': list(set(active))})

    # ── 2. PREGNANCY EXCEL SOURCE ────────────────────────────────────────────
    p2 = os.path.join(DATA_DIR, "Animal_Vet_Pregnancy.xlsx")
    if os.path.exists(p2):
        df_preg = pd.read_excel(p2)
        df_preg_pos = df_preg[df_preg['Pregnancy_Status'].astype(str).str.strip().str.lower() == 'yes']
        df_preg_pos = df_preg_pos.sample(n=min(25, len(df_preg_pos)), random_state=42)
        
        for _, row in df_preg_pos.iterrows():
            species = str(row.get('Species', 'Dog')).strip()
            temp_f = pd.to_numeric(row.get('Body_Temperature_F'), errors='coerce')
            temp = round((temp_f - 32) * 5 / 9, 2) if not pd.isna(temp_f) else HEALTHY_VITALS.get(species, {'temp':38.5})['temp']
            hr = HEALTHY_VITALS.get(species, {'hr':85})['hr']
            
            active = []
            if str(row.get('Vomiting','')).strip().lower() == 'yes': active.append('Vomiting')
            
            app_change = str(row.get('Appetite_Change','')).strip().lower()
            if app_change == 'increased': active.append('Increased Appetite')
            elif app_change == 'decreased': active.append('Appetite Loss')
                
            beh_change = str(row.get('Behavior_Change','')).strip().lower()
            if 'nesting' in beh_change: active.append('Nesting Behavior')
            elif 'restless' in beh_change: active.append('Restless Behavior')
            elif 'aggressive' in beh_change: active.append('Aggressive Behavior')
            elif 'lethargic' in beh_change: active.append('Lethargy')
                
            disc_type = str(row.get('Discharge_Type','')).strip().lower()
            if disc_type == 'clear': active.append('Clear Vaginal Discharge')
            elif disc_type == 'bloody': active.append('Bloody Vaginal Discharge')
            elif disc_type == 'purulent': active.append('Purulent Vaginal Discharge')
                
            if str(row.get('Fetal_Heart_Sound','')).strip().lower() == 'detected':
                active.append('Fetal Heart Sound Detected')
                
            records.append({'species': species, 'disease': 'Pregnancy', 'temp': float(temp), 'hr': float(hr), 'symptoms': list(set(active))})

    # ── 3. VET TEXT DATASET (NLP TRANSLATION BRIDGE) ─────────────────────────
    p3 = os.path.join(DATA_DIR, "pet-health-symptoms-dataset.csv")
    if os.path.exists(p3):
        df_ph = pd.read_csv(p3)
        for _, row in df_ph.iterrows():
            text_str = str(row.get('text', '')).strip().lower()
            condition_tag = str(row.get('condition', '')).strip()
            record_type = str(row.get('record_type', '')).strip()
            
            # Extract specific symptom matches from the raw text using synonym rules
            matched_symptoms = extract_symptoms_from_text(text_str)
            
            # Also map the condition_tag as a fallback if no synonyms matched
            mapped_base = SYMPTOM_MAP.get(condition_tag.lower(), condition_tag)
            if not matched_symptoms and mapped_base:
                matched_symptoms = [mapped_base]

            # Populate NLP text corpus with full symptom list per document
            text_corpus.append(text_str)
            text_labels.append(matched_symptoms)
            
            # Only clinical notes are trusted to train the Random Forest
            if record_type != "Clinical Notes": continue
            
            species = 'Dog'
            if 'cat' in text_str or 'feline' in text_str: species = 'Cat'
            elif 'horse' in text_str or 'equine' in text_str: species = 'Horse'
            
            # Use the base mapped disease label for disease training
            cleaned_disease = clean_disease_name(mapped_base)
            records.append({
                'species': species, 'disease': cleaned_disease,
                'temp': HEALTHY_VITALS.get(species, {'temp':38.5})['temp'],
                'hr': HEALTHY_VITALS.get(species, {'hr':85})['hr'],
                'symptoms': list(set(matched_symptoms))
            })

    # ── 4. CONTROL BASELINES FOR HEALTHY ANIMALS ─────────────────────────────
    for sp, vitals in HEALTHY_VITALS.items():
        for _ in range(30):
            records.append({'species': sp, 'disease': 'Healthy / No Disease', 'temp': vitals['temp'], 'hr': vitals['hr'], 'symptoms': []})

    # ── 5. SYNONYM DOCUMENT INJECTION ────────────────────────────────────────
    # Inject one synthetic document per symptom so the TF-IDF vectorizer has
    # perfect signal for any synonym query. This guarantees cosine > 0.3 for
    # direct synonym searches that might not appear in the real notes corpus.
    print("Injecting synonym anchor documents into NLP corpus...")
    for sym, keywords in SYNONYM_RULES.items():
        anchor_text = f"{sym.lower()} " + " ".join(keywords)
        text_corpus.append(anchor_text)
        text_labels.append([sym])

    # --- TF-IDF VECTORIZER ---
    print("Fitting text extraction vectorizer matrix...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=3000)
    tfidf_matrix = vectorizer.fit_transform(text_corpus)
    
    joblib.dump(vectorizer, os.path.join(OUTPUT_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(tfidf_matrix, os.path.join(OUTPUT_DIR, "tfidf_matrix.joblib"))
    with open(os.path.join(OUTPUT_DIR, "text_labels.json"), "w") as f:
        json.dump(text_labels, f)

    # --- MODEL CORE MACHINE LEARNING BRAIN ---
    df_master = pd.DataFrame(records)
    le_sp = LabelEncoder()
    df_master['species_enc'] = le_sp.fit_transform(df_master['species'])
    
    mlb = MultiLabelBinarizer()
    sym_enc = mlb.fit_transform(df_master['symptoms'])
    sym_cols = [f"sym_{s}" for s in mlb.classes_]
    df_sym = pd.DataFrame(sym_enc, columns=sym_cols, index=df_master.index)
    
    X = pd.concat([df_master[['species_enc', 'temp', 'hr']], df_sym], axis=1)
    le_dis = LabelEncoder()
    y = le_dis.fit_transform(df_master['disease'])
    
    print(f"Training on {len(df_master)} records with {len(mlb.classes_)} symptoms and {len(le_dis.classes_)} diseases...")
    rf = RandomForestClassifier(
        n_estimators=300, 
        max_depth=30, 
        min_samples_split=5, 
        min_samples_leaf=2, 
        random_state=42, 
        n_jobs=-1
    )
    rf.fit(X, y)
    
    from sklearn.model_selection import cross_val_score
    score = round(cross_val_score(rf, X, y, cv=3).mean() * 100, 2)
    print(f"Verified Model Generalization Score: {score}%")
    joblib.dump(rf, os.path.join(OUTPUT_DIR, "model.joblib"))
    
    encoders_json = {
        "species": le_sp.classes_.tolist(),
        "disease": le_dis.classes_.tolist(),
        "symptoms": mlb.classes_.tolist(),
        "feature_names": X.columns.tolist(),
        "healthy_vitals": HEALTHY_VITALS,
        "synonym_rules": SYNONYM_RULES  # Export for unified use in main.py
    }
    with open(os.path.join(OUTPUT_DIR, "encoders.json"), "w") as f:
        json.dump(encoders_json, f, indent=2)

    # --- DISEASE ENCYCLOPEDIA INFO ---
    info_map = {}
    p4 = os.path.join(DATA_DIR, "Animal disease spreadsheet - Sheet1.csv")
    if os.path.exists(p4):
        df_info = pd.read_csv(p4)
        for _, row in df_info.iterrows():
            name = row.get('Unnamed: 0')
            if pd.isna(name): continue
            
            desc = str(row.get('Description', '')).strip()
            treat = str(row.get('Treatment', '')).strip()
            prev = str(row.get('Advice/ Prevention', '')).strip()
            
            # Clean values
            if desc.lower() in ('nan', 'none', 'null', ''):
                desc = 'Clinical medical condition.'
            if treat.lower() in ('nan', 'none', 'null', ''):
                treat = 'Consult a veterinarian for detailed treatment and diagnosis.'
            if prev.lower() in ('nan', 'none', 'null', ''):
                prev = 'Maintain general hygiene, follow regular vaccination schedules, and schedule routine veterinary checkups.'
                
            info_map[clean_disease_name(name)] = {
                'description': desc,
                'treatment': treat,
                'prevention': prev,
            }
            
    # Add safety fallbacks for any disease not in encyclopedia, and clean any existing values
    for d in encoders_json["disease"]:
        if d not in info_map:
            info_map[d] = {
                'description': 'Clinical medical condition.',
                'treatment': 'Consult a veterinarian for detailed systemic tracking.',
                'prevention': 'Maintain diagnostic tracking and clear habitat management protocols.'
            }
        else:
            entry = info_map[d]
            desc = str(entry.get('description', '')).strip()
            treat = str(entry.get('treatment', '')).strip()
            prev = str(entry.get('prevention', '')).strip()
            
            if desc.lower() in ('nan', 'none', 'null', ''):
                desc = 'Clinical medical condition.'
            if treat.lower() in ('nan', 'none', 'null', ''):
                treat = 'Consult a veterinarian for detailed treatment and diagnosis.'
            if prev.lower() in ('nan', 'none', 'null', ''):
                prev = 'Maintain general hygiene, follow regular vaccination schedules, and schedule routine veterinary checkups.'
                
            info_map[d] = {
                'description': desc,
                'treatment': treat,
                'prevention': prev,
            }
            
    with open(os.path.join(OUTPUT_DIR, "disease_info.json"), "w") as f:
        json.dump(info_map, f, indent=2)
    
    print(f"\nPre-training complete!")
    print(f"  Symptoms: {len(mlb.classes_)}")
    print(f"  Diseases: {len(le_dis.classes_)}")
    print(f"  Species:  {len(le_sp.classes_)}")
    print(f"  Corpus:   {len(text_corpus)} documents")
    print(f"  Score:    {score}%")

if __name__ == "__main__":
    build_and_train()