import polars as pl
import sys
import json
from tqdm import tqdm

labelmap = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
split = sys.argv[3]

products = (
    pl.read_parquet(sys.argv[1])
    .filter((pl.col("product_locale") == "us"))
    .with_columns(
        pl.concat_str(
            [
                pl.col("product_title"),
                pl.col("product_description"),
                pl.col("product_bullet_point"),
                pl.col("product_brand"),
                pl.col("product_color"),
            ],
            separator=" ",
            ignore_nulls=True,
        ).alias("text")
    )
    .select(pl.col("product_id", "text"))
)
examples = (
    pl.read_parquet(sys.argv[2])
    .filter(
        (pl.col("product_locale") == "us")
        & (pl.col("small_version") == 1)
        & (pl.col("split") == split)
    )
    .with_columns(
        pl.col("esci_label").replace(labelmap).alias("score").cast(pl.Float64)
    )
    .select(pl.col("query_id", "query", "product_id", "score"))
)
merged = examples.join(products, on="product_id", how="left")

print(merged)

result = merged.group_by("query_id").agg(
    pl.first("query"), pl.col("text"), pl.col("score")
)


def save_json(df: pl.DataFrame, path: str):
    with open(path, "w") as f:
        for row in tqdm(result.to_dicts(), desc=f"saving {path}"):
            query = row["query"]
            pos = []
            neg = []
            negscore = []
            for doc, score in zip(row["text"], row["score"]):
                if score == 1.0:
                    pos.append(doc)
                else:
                    neg.append(doc)
                    negscore.append(score)
            for p in pos:
                line = json.dumps(
                    {"query": query, "doc": p, "neg": neg, "negscore": negscore}
                )
                f.write(line + "\n")


save_json(result, f"{split}.jsonl")
