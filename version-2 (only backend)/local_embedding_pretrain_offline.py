"""
local_embedding_pretrain_offline.py
────────────────────────────────────
Offline version of the embedding pre-calculator.
Uses the sentence-transformers library directly (runs the model locally)
instead of calling the HF inference API.

This is ideal when the machine running this script doesn't have outbound
access to api-inference.huggingface.co, but CAN pip-install packages.

The model (all-MiniLM-L6-v2) is ~90MB and will be downloaded from HuggingFace
Hub the first time, then cached locally. Subsequent runs are instant.

Run:
    pip install sentence-transformers
    python local_embedding_pretrain_offline.py
"""

import os
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Master symptom vocabulary ─────────────────────────────────────────────────
# Same SYMPTOM_PHRASES as the API version — every canonical symptom gets
# its canonical name + several natural language paraphrases so the model
# can understand conversational user input like "won't eat" -> Appetite Loss.
SYMPTOM_PHRASES: dict[str, list[str]] = {
    "Appetite Loss": [
        "not eating", "won't eat", "refuse food", "refusing food", "loss of appetite",
        "decreased appetite", "off food", "stopped eating", "no interest in food",
        "not touching food", "doesn't want to eat",
    ],
    "Increased Appetite": [
        "always hungry", "eating more than usual", "voracious appetite", "polyphagia",
        "constantly begging for food", "never full",
    ],
    "Coughing": [
        "coughing", "dry cough", "hacking cough", "persistent cough",
        "honking cough", "kennel cough", "keeps coughing",
    ],
    "Sneezing": [
        "sneezing constantly", "reverse sneezing", "sneezes a lot",
        "frequent sneezing", "nasal discharge with sneezing",
    ],
    "Nasal Discharge": [
        "runny nose", "nasal mucus", "snot from nose", "discharge from nostrils",
        "congested nose", "nose dripping", "green nasal discharge", "yellow nasal discharge",
    ],
    "Eye Discharge": [
        "runny eyes", "watery eyes", "goopy eyes", "crusty eyes",
        "discharge from eyes", "eye gunk", "tearing excessively",
    ],
    "Labored Breathing": [
        "difficulty breathing", "breathing hard", "panting", "wheezing",
        "shortness of breath", "respiratory distress", "gasping", "open-mouth breathing",
        "heavy breathing", "rapid breathing",
    ],
    "Fever": [
        "high temperature", "feels hot to touch", "feverish", "running a fever",
        "burning up", "warm to touch", "high temp",
    ],
    "Lethargy": [
        "very tired", "no energy", "sluggish", "lethargic", "depressed",
        "inactive", "listless", "dull and slow", "sleeping all day", "weak and tired",
    ],
    "Vomiting": [
        "throwing up", "vomiting", "vomits", "puking", "retching",
        "bringing up food", "regurgitating", "sick to stomach",
    ],
    "Diarrhea": [
        "loose stool", "watery poop", "diarrhea", "diarrhoea", "runny stool",
        "liquid stool", "bloody stool", "scours",
    ],
    "Dehydration": [
        "dehydrated", "dry gums", "sunken eyes", "not drinking water",
        "tacky gums", "skin tenting", "very thirsty",
    ],
    "Weight Loss": [
        "losing weight", "getting thinner", "emaciated", "wasting away",
        "thin body", "bony", "ribs showing",
    ],
    "Skin Lesions": [
        "skin sores", "rash", "scabs on skin", "crusty patches",
        "skin bumps", "wounds on skin", "lesions", "open sores",
    ],
    "Itching / Scratching": [
        "scratching all the time", "itchy skin", "biting own skin",
        "chewing paws", "rubbing face on floor", "intense itching",
    ],
    "Hair Loss": [
        "losing fur", "bald spots", "alopecia", "patchy coat",
        "thinning coat", "fur falling out", "missing hair",
    ],
    "Lameness": [
        "limping", "favoring one leg", "difficulty walking", "cannot walk properly",
        "lame", "not bearing weight", "stiff legged", "hobbling",
    ],
    "Swelling": [
        "swollen area", "lump on body", "bump", "abscess", "swelled up",
        "growth under skin", "puffy",
    ],
    "Swollen Joints": [
        "joints are swollen", "joint pain", "stiff joints", "arthritis signs",
        "joint swelling", "hot joints",
    ],
    "Swollen Legs": [
        "puffy legs", "swollen legs", "edema in legs", "leg swelling",
    ],
    "Weakness / Stiffness": [
        "muscle weakness", "wobbly", "unstable", "trembling", "shaking",
        "shivering", "cannot stand up", "collapsing", "stiff muscles",
    ],
    "Digestive Issues": [
        "bloated stomach", "abdominal pain", "gas", "stomach ache",
        "indigestion", "gurgling stomach", "distended abdomen",
    ],
    "Excessive Drooling": [
        "drooling excessively", "hypersalivation", "slobbering",
        "saliva dripping", "mouth foaming",
    ],
    "Parasites": [
        "fleas", "ticks", "worms", "lice", "mites", "intestinal parasites",
        "roundworms", "tapeworms",
    ],
    "Ear Infections": [
        "ear infection", "scratching ears", "smelly ears", "dark ear wax",
        "head shaking", "discharge from ears", "ear odor",
    ],
    "Restless Behavior": [
        "pacing", "cannot settle", "agitated", "restless", "pacing back and forth",
        "cannot rest", "anxious movement",
    ],
    "Nesting Behavior": [
        "nesting", "building a nest", "gathering bedding", "making a nest",
        "preparing a den",
    ],
    "Aggressive Behavior": [
        "aggressive", "growling", "biting", "snapping", "lunging",
        "hostile", "vicious", "attacking",
    ],
    "Clear Vaginal Discharge": [
        "clear discharge from vagina", "mucus vaginal discharge",
        "vaginal discharge", "clear vaginal",
    ],
    "Bloody Vaginal Discharge": [
        "bloody vaginal discharge", "vaginal bleeding", "blood from vagina",
        "bloody discharge",
    ],
    "Purulent Vaginal Discharge": [
        "pus from vagina", "green vaginal discharge", "yellow smelly discharge",
        "infected vaginal discharge",
    ],
    "Fetal Heart Sound Detected": [
        "fetal heartbeat detected", "puppy heartbeat", "fetal movement",
        "heart sound in abdomen",
    ],
    "Decreased Milk Yield": [
        "less milk", "milk production dropped", "dry milk", "low milk output",
        "milk supply decreased",
    ],
    "Reduced Wool Production": [
        "less wool", "poor fleece", "thin fleece", "reduced wool",
        "wool drop",
    ],
}


def build_phrase_corpus(symptom_phrases: dict) -> tuple[list[str], list[str]]:
    phrases, labels = [], []
    for canonical, phrase_list in symptom_phrases.items():
        phrases.append(canonical.lower())
        labels.append(canonical)
        for p in phrase_list:
            phrases.append(p.lower())
            labels.append(canonical)
    return phrases, labels


def main():
    print("=" * 60)
    print("  Pet Vet AI -- Offline Embedding Pre-calculator")
    print("  Model  : sentence-transformers/all-MiniLM-L6-v2")
    print(f"  Output : {OUTPUT_DIR}")
    print("=" * 60)

    print("\n[INFO] Loading model (will download ~90MB on first run)...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("[OK] Model loaded.")

    phrases, labels = build_phrase_corpus(SYMPTOM_PHRASES)
    print(f"[INFO] Corpus: {len(phrases)} phrases across {len(SYMPTOM_PHRASES)} symptoms")

    print("[INFO] Encoding all phrases locally...")
    embeddings = model.encode(
        phrases,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2-normalize for cosine = dot product (faster)
        convert_to_numpy=True,
    ).astype(np.float32)

    print(f"[OK] Embeddings shape: {embeddings.shape}")

    # -- Save -----------------------------------------------------------------
    emb_path   = os.path.join(OUTPUT_DIR, "symptom_embeddings.joblib")
    label_path = os.path.join(OUTPUT_DIR, "symptom_labels.joblib")

    joblib.dump(embeddings, emb_path,   compress=3)
    joblib.dump(labels,     label_path, compress=3)

    print(f"\n[SAVED]")
    print(f"   {emb_path}   ({os.path.getsize(emb_path) // 1024} KB)")
    print(f"   {label_path} ({os.path.getsize(label_path) // 1024} KB)")

    # -- Quick sanity check ---------------------------------------------------
    from sklearn.metrics.pairwise import cosine_similarity
    tests = [
        "throwing up and vomiting",
        "my dog won't eat",
        "horse is limping badly",
        "cat is sneezing and has watery eyes",
    ]
    print("\n[TEST] Sanity checks:")
    for t in tests:
        tvec = model.encode([t], normalize_embeddings=True).astype(np.float32)
        sims = cosine_similarity(tvec, embeddings)[0]
        top  = sorted(set(labels[i] for i in np.argsort(sims)[::-1][:5]), key=lambda x: -sims[labels.index(x)])[:3]
        print(f"  '{t}'")
        print(f"    -> {', '.join(top)}")

    print("\n[DONE] Pre-calculation complete! Assets ready for deployment.\n")


if __name__ == "__main__":
    main()
