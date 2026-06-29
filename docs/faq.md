# Frequently Asked Questions

### 1. Can Tidely connect directly to my SQL Database?
Not yet. Tidely currently expects a DataFrame as input. You should query your database using Pandas (`pd.read_sql()`) or Polars, and then pass the resulting DataFrame into `td.clean()`.

### 2. Does Tidely use a Large Language Model (LLM) under the hood?
No. Tidely is 100% deterministic, meaning it uses strict code paths and RegEx inference engines rather than neural nets. It will never hallucinate transformations.

### 3. Does Tidely delete missing values (NaNs)?
No. Tidely strongly believes that dropping missing values without the user's permission is a dangerous anti-pattern. Missing values are either safely left alone (generating a `Warning`), or imputed *only* if doing so carries a 100% safety guarantee.

### 4. How does the Trust Score work?
The Trust Score evaluates 5 dimensions of your dataset:
- **Reliability**: How many nulls/duplicates exist?
- **ML Readiness**: Are the types strictly typed and encoded correctly?
- **Memory Efficiency**: Is the precision downcasted?
- **Schema Stability**: Are there structural anomalies?
- **Semantic Quality**: Did the dataset pass semantic inference checks?

The score naturally increases significantly after running `td.clean()`.
