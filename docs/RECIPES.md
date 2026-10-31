# Recipes for other engines

These engines need no adapter: use `CleanerHandle` (one cleaner per process, rebuilt after a fork, pickles as config only) and `clean_columns` (cleans listed columns of a `{column: list}` batch). Same `columns` / `output` / `config` rules as in [INTEGRATIONS.md](INTEGRATIONS.md).

```python
from sct.integrations._runtime import CleanerHandle, clean_columns

handle = CleanerHandle()            # "training" preset
clean_columns(handle, {"text": ["mail john@example.com"]})
# {'text': ['mail <EMAIL>']}
```

Polars and DuckDB were run by hand against SCT 0.8.0. The others follow the engine's documented batch API and are untested.

## Polars

```python
import polars as pl

handle = CleanerHandle()
df = pl.DataFrame({"text": ["mail john@example.com", None]})
df = df.with_columns(
    pl.col("text").map_batches(
        lambda s: pl.Series(clean_columns(handle, {"text": s.to_list()})["text"]),
        return_dtype=pl.String,
    )
)
```

## DuckDB

```python
import duckdb
import pyarrow as pa

handle = CleanerHandle()

def sct_clean(texts: pa.Array) -> pa.Array:
    return pa.array(clean_columns(handle, {"text": texts.to_pylist()})["text"], pa.string())

con = duckdb.connect()
con.create_function("sct_clean", sct_clean, ["VARCHAR"], "VARCHAR", type="arrow")
con.sql("SELECT sct_clean(text) AS text FROM read_json_auto('raw.jsonl')")
```

## Ray Data

```python
import ray

class Clean:
    def __init__(self):
        self.handle = CleanerHandle()

    def __call__(self, batch):
        return clean_columns(self.handle, {k: list(v) for k, v in batch.items()})

ds = ray.data.read_json("raw/").map_batches(Clean, batch_format="numpy", batch_size=64, concurrency=4)
```

## Spark

```python
handle = CleanerHandle()   # shipped to executors as config only

def clean_partition(rows):
    rows = list(rows)
    texts = clean_columns(handle, {"text": [r["text"] for r in rows]})["text"]
    for row, text in zip(rows, texts):
        yield {**row.asDict(), "text": text}

clean = spark.read.json("raw/").rdd.mapPartitions(clean_partition).toDF()
```

Spark deserializes the closure per task, so each partition loads the models again: use few, large partitions.

## dlt

```python
import dlt

handle = CleanerHandle()

def clean_row(row):
    row["text"] = clean_columns(handle, {"text": [row["text"]]})["text"][0]
    return row

source = my_resource().add_map(clean_row)   # one row per call; batch upstream for speed
dlt.pipeline("clean", destination="duckdb").run(source)
```

## Daft

```python
import daft

@daft.udf(return_dtype=daft.DataType.string())
class Clean:
    def __init__(self):
        self.handle = CleanerHandle()

    def __call__(self, texts):
        return clean_columns(self.handle, {"text": texts.to_pylist()})["text"]

df = daft.read_json("raw/").with_column("text", Clean(daft.col("text")))
```
