import json
import spacy
from pathlib import Path

# Lade spaCy NER-Modell (englisch)
nlp = spacy.load("en_core_web_sm")

# Definiere die Maskierungs-Policy
POLICY = {
    "EMPLOYER": {"PERSON": True, "ORG": True, "GPE": True, "DATE": True},
    "EMPLOYEE": {"PERSON": False, "ORG": True, "GPE": False, "DATE": True},
    "CUSTOMER": {"PERSON": False, "ORG": False, "GPE": False, "DATE": True}
}

# Mapping für Maskierungstokens
MASK_MAP = {
    "PERSON": "[PERSON]",
    "ORG": "[ORG]",
    "GPE": "[LOCATION]",
    "DATE": "[DATE]"
}


def mask_text(text: str, role: str) -> str:
    doc = nlp(text)
    allowed = POLICY.get(role, {})
    masked_text = text
    offset = 0

    for ent in doc.ents:
        label = ent.label_
        if label in allowed and not allowed[label]:
            start = ent.start_char + offset
            end = ent.end_char + offset
            mask_token = MASK_MAP.get(label, f"[{label}]")
            masked_text = masked_text[:start] + mask_token + masked_text[end:]
            offset += len(mask_token) - (end - start)

    return masked_text


def process_squad(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    count = 0
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for article in data["data"]:
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]
                for qa in paragraph["qas"]:
                    if not qa.get("answers") and qa.get("is_impossible", False):
                        continue  # Skip questions without answers

                    question = qa["question"]
                    id_ = qa["id"]
                    answers = qa.get("answers", [])
                    first_answer = answers[0]["text"] if answers else ""

                    role_versions = {}
                    role_answerable = {}

                    for role in POLICY.keys():
                        masked_context = mask_text(context, role)

                        role_versions[role] = masked_context

                        # Simple heuristic: is the answer still findable in the masked context?
                        role_answerable[role] = first_answer in masked_context

                    all_masked_same = all(
                        role_versions[role] == context
                        for role in POLICY if role != "Chef"
                    )

                    if all_masked_same:
                        continue

                    new_entry = {
                        "id": id_,
                        "question": question,
                        "original_context": context,
                        "original_answer": first_answer,
                        "role_contexts": role_versions,
                        "role_answerable": role_answerable
                    }
                    count = count + 1
                    f.write(json.dumps(new_entry, ensure_ascii=False) + "\n")
                    if(count > 9999):
                        return

    print(f"✅ Processing done {output_path}")


BASE_DIR = Path(__file__).resolve().parent.parent
input_path = BASE_DIR / "role-aware-rag" / "data" / "squad.json"
output_path = BASE_DIR / "role-aware-rag" / "output" / "role_aware_squad.json"

if __name__ == "__main__":
    process_squad(input_path, output_path)
