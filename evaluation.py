import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np

def normalize(text):
    if pd.isna(text):
        return ""
    return str(text).strip().lower()


def normalize_answers(df):
    df["llm_answer_norm"] = df["llm_answer"].apply(normalize)
    df["original_answer_norm"] = df["original_answer"].apply(normalize)
    df["answers_match"] = df.apply(lambda row: row["original_answer_norm"] in row["llm_answer_norm"], axis=1)
    df["exact_match"] = df["original_answer_norm"] == df["llm_answer_norm"]

    return df

def compute_vanilla_metrics(df):

    # Normalize answers
    df = normalize_answers(df)

    results = []

    df["gold"] = df["original_answer_norm"].apply(lambda x: bool(x and x.strip()))
    df["pred"] = df.apply(
        lambda row: row["llm_says_answerable"] and row["answers_match"], axis=1
    )


    y_true = df["gold"].astype(bool)
    y_pred = df["pred"].astype(bool)

    tp = ((y_true) & (y_pred))
    fp = ((~y_true) & (y_pred))
    fn = ((y_true) & (~y_pred))
    tn = ((~y_true) & (~y_pred))

    # Classification metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=True, zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)

    tp_rows = df[tp]
    em_accuracy = tp_rows["exact_match"].mean() if len(tp_rows) else np.nan

    results.append({
        "n_heuristic_answerable": int(y_true.sum()),
        "n_llm_answerable": int(y_pred.sum()),
        "true_positives": int(tp.sum()),
        "false_positives": int(fp.sum()),
        "false_negatives": int(fn.sum()),
        "true_negatives": int(tn.sum()),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": acc,
        "answer_exact_match_accuracy": em_accuracy
    })

    return pd.DataFrame(results)

def compute_final_metrics(df):

    if "model" not in df.columns:
        df["model"] = "default"

    df = normalize_answers(df)

    results = []

    for (model, role), group in df.groupby(["model", "role"]):
        y_true = group["is_heuristically_answerable"]
        y_pred = group["answers_match"]

        # Classification metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", pos_label=True, zero_division=0
        )
        acc = accuracy_score(y_true, y_pred)

        # Exact match accuracy on correctly predicted answerables
        answerable = group[(group["answers_match"] == True) & (group["is_heuristically_answerable"] == True)]
        em_accuracy = answerable["exact_match"].mean() if len(answerable) > 0 else None

        results.append({
            "model": model,
            "role": role,
            "n_total": len(group),
            "n_heuristic_answerable": int(y_true.sum()),
            "n_llm_answerable": int(y_pred.sum()),
            "true_positives": int(((y_pred == True) & (y_true == True)).sum()),
            "false_positives": int(((y_pred == True) & (y_true == False)).sum()),
            "false_negatives": int(((y_pred == False) & (y_true == True)).sum()),
            "true_negatives": int(((y_pred == False) & (y_true == False)).sum()),
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": acc,
            "answer_exact_match_accuracy": em_accuracy,
        })

    return pd.DataFrame(results)


pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)  # don't truncate cell values


#df = pd.read_csv("prompt_engineering.csv")
#df = compute_final_metrics(df)
#df.to_csv("results_prompt_engineering.csv", index=False)

#df = pd.read_csv("llama.csv")
#df = compute_final_metrics(df)
#df.to_csv("results_llama.csv")
#print(df)

# Usage example
df_engineering = pd.read_csv("prompt_engineering.csv")
#df_vanilla = pd.read_csv("vanilla.csv")
results_engineering_df = compute_final_metrics(df_engineering)
#results_vanilla_df = compute_vanilla_metrics(df_vanilla)
# Display or export
print(results_engineering_df)
#print(results_vanilla_df)

results_engineering_df.to_csv("results_prompt_engineering.csv", index=False)
#results_vanilla_df.to_csv("results_vanilla.csv", index=False)




