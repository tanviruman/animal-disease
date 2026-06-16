
import os
import time
import json
import joblib
import requests
import numpy as np

HF_TOKEN   = os.getenv("HF_TOKEN", "")
MODEL_ID   = "sentence-transformers/all-MiniLM-L6-v2"
API_URL    = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{MODEL_ID}"
HEADERS    = {"Authorization": f"Bearer {HF_TOKEN}"}
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
BATCH_SIZE = 64   # HF free tier handles up to 100 inputs per call safely

os.makedirs(OUTPUT_DIR, exist_ok=True)


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


def call_hf_api(texts: list[str], retries: int = 3) -> list[list[float]]:
    """
    Calls the HF free inference API and returns a list of 384-dim float vectors.
    Retries up to `retries` times on 503 (model loading) with exponential backoff.
    """
    for attempt in range(retries):
        resp = requests.post(API_URL, headers=HEADERS, json={"inputs": texts}, timeout=60)

        if resp.status_code == 503:
            wait = 20 * (attempt + 1)
            print(f"  ⏳ API warming up (503). Waiting {wait}s…")
            time.sleep(wait)
            continue

        if resp.status_code == 200:
            data = resp.json()
            # HF returns either [[vec], [vec]] or [vec] for single input
            if isinstance(data[0], list) and isinstance(data[0][0], float):
                return data          # already a list of vectors
            # Some model versions wrap in an extra list
            return [row[0] if isinstance(row[0], list) else row for row in data]

        raise RuntimeError(f"HF API error {resp.status_code}: {resp.text[:300]}")

    raise RuntimeError("HF API failed after maximum retries.")


def build_phrase_corpus(symptom_phrases: dict) -> tuple[list[str], list[str]]:
    """
    Flatten symptom_phrases into parallel lists:
        phrases  — the text to embed (e.g. "throwing up", "not eating")
        labels   — the canonical symptom for each phrase (e.g. "Vomiting")

    We also add the canonical name itself as a phrase so it matches verbatim.
    """
    phrases, labels = [], []
    for canonical, phrase_list in symptom_phrases.items():
        # Add the canonical name as a high-signal phrase
        phrases.append(canonical.lower())
        labels.append(canonical)
        for p in phrase_list:
            phrases.append(p.lower())
            labels.append(canonical)
    return phrases, labels


def embed_in_batches(phrases: list[str]) -> np.ndarray:
    """Splits phrases into BATCH_SIZE chunks, embeds each, returns stacked numpy array."""
    all_vectors = []
    total = len(phrases)
    for i in range(0, total, BATCH_SIZE):
        batch = phrases[i : i + BATCH_SIZE]
        print(f"  Embedding batch {i // BATCH_SIZE + 1} / {(total + BATCH_SIZE - 1) // BATCH_SIZE}"
              f"  ({len(batch)} phrases)…")
        vecs = call_hf_api(batch)
        all_vectors.extend(vecs)
        time.sleep(0.5)   # polite rate-limiting for free tier
    return np.array(all_vectors, dtype=np.float32)


def main():
    print("=" * 60)
    print("  Pet Vet AI -- Semantic Embedding Pre-calculator")
    print(f"  Model  : {MODEL_ID}")
    print(f"  Output : {OUTPUT_DIR}")
    print("=" * 60)

    phrases, labels = build_phrase_corpus(SYMPTOM_PHRASES)
    print(f"\n[INFO] Corpus: {len(phrases)} phrases across {len(SYMPTOM_PHRASES)} symptoms\n")

    print("[INFO] Connecting to Hugging Face Inference API...")
    embeddings = embed_in_batches(phrases)

    print(f"\n[OK] Got embeddings: shape = {embeddings.shape}")

    # -- Save -----------------------------------------------------------------
    emb_path   = os.path.join(OUTPUT_DIR, "symptom_embeddings.joblib")
    label_path = os.path.join(OUTPUT_DIR, "symptom_labels.joblib")

    joblib.dump(embeddings, emb_path,   compress=3)
    joblib.dump(labels,     label_path, compress=3)

    print(f"\n[SAVED]")
    print(f"   {emb_path}  ({os.path.getsize(emb_path) // 1024} KB)")
    print(f"   {label_path}  ({os.path.getsize(label_path) // 1024} KB)")
    print("\n[DONE] Pre-calculation complete! Deploy these assets with your API.\n")

    # -- Quick sanity check ---------------------------------------------------
    from sklearn.metrics.pairwise import cosine_similarity
    test_text = ["throwing up and vomiting"]
    test_vec  = np.array(call_hf_api(test_text), dtype=np.float32)
    sims      = cosine_similarity(test_vec, embeddings)[0]
    top_idx   = np.argsort(sims)[::-1][:3]
    print("[TEST] Sanity check -- 'throwing up and vomiting' top matches:")
    for i in top_idx:
        print(f"   {labels[i]:<35} {sims[i]:.4f}")


if __name__ == "__main__":
    main()
