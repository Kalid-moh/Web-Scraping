from db.connection import get_connection
import pandas as pd
from psycopg2.extras import execute_batch

# Load dataset
SENTIMENT_CSV = r"C:\Users\h\Desktop\WEEK 2\Web-Scraping\data\processed\reviews_with_sentiment_themes.csv"

df = pd.read_csv(SENTIMENT_CSV)

# Connect to DB
conn = get_connection()
cur = conn.cursor()

# Get unique banks from your dataset
banks = df["bank"].unique()

for b in banks:
    cur.execute("INSERT INTO banks (bank_name) VALUES (%s) ON CONFLICT DO NOTHING;", (b,))

conn.commit()

cur.execute("SELECT bank_id, bank_name FROM banks;")
rows = cur.fetchall()
bank_map = {name: bank_id for bank_id, name in rows}

# Add bank_id column to df
df["bank_id"] = df["bank"].map(bank_map)

# Prepare DataFrame for review insertion
df = df.rename(columns={
    "ratin": "rating",
    "date": "review_date",
    "sentiment_compound": "sentiment_score"
})


df_db = df[[
    "bank_id",
    "review_text",
    "rating",
    "review_date",
    "source",
    "sentiment_label",
    "sentiment_score",
    "theme_label"
]]

df_db["review_date"] = pd.to_datetime(df_db["review_date"])

# Insert reviews into PostgreSQL
query = """
INSERT INTO reviews
(bank_id, review_text, rating, review_date, source, sentiment_label, sentiment_score, theme_label)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s);
"""

data = [tuple(row) for row in df_db.to_numpy()]
execute_batch(cur, query, data, page_size=500)

conn.commit()
cur.close()
conn.close()

print(f"Inserted {len(df_db)} reviews successfully!")

