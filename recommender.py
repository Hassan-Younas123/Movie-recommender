import sys
import difflib

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def load_and_clean(path):
    df = pd.read_csv(path).drop_duplicates()
    df["year"] = pd.to_numeric(df["title"].str.extract(r"\((\d{4})\)\s*$")[0], errors="coerce")
    df["genres"] = df["genres"].replace("(no genres listed)", "")
    df["genre_list"] = df["genres"].apply(lambda g: [x for x in g.split("|") if x])
    df["n_genres"] = df["genre_list"].str.len()
    df["decade"] = (df["year"] // 10 * 10).astype("Int64")
    df["decade_tok"] = df["decade"].apply(lambda d: f"decade_{d}" if pd.notna(d) else "")
    df["soup"] = df.apply(
        lambda r: " ".join([g.replace("-", "") for g in r["genre_list"]] + [r["decade_tok"]]).strip(),
        axis=1,
    )
    df["lower_title"] = df["title"].str.lower()
    return df.reset_index(drop=True)


def eda(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print(f"Movies: {len(df):,} | missing year: {df['year'].isna().sum():,} | "
          f"no genres: {(df['n_genres'] == 0).sum():,} | dup titles: {df['title'].duplicated().sum():,}")
    print(df.explode("genre_list")["genre_list"].value_counts().head(10))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    df.explode("genre_list")["genre_list"].value_counts().head(15)[::-1].plot.barh(ax=ax[0], title="Top genres")
    df["year"].dropna().plot.hist(bins=40, ax=ax[1], title="Movies by year")
    df["n_genres"].value_counts().sort_index().plot.bar(ax=ax[2], title="Genres per movie")
    plt.tight_layout()
    plt.savefig("eda.png", dpi=120)


class Recommender:
    def __init__(self, df):
        self.df = df[df["soup"].str.len() > 0].reset_index(drop=True)
        self.tfidf = TfidfVectorizer(token_pattern=r"[A-Za-z0-9_]+")
        self.X = self.tfidf.fit_transform(self.df["soup"])
        self._titles = self.df["lower_title"].tolist()

    def find(self, query):
        q = query.lower().strip()
        if q in self._titles:
            return self._titles.index(q)
        sub = [i for i, t in enumerate(self._titles) if q in t]
        if sub:
            return sub[0]
        m = difflib.get_close_matches(q, self._titles, n=1, cutoff=0.6)
        return self._titles.index(m[0]) if m else None

    def recommend(self, title, n=10, min_year=None):
        idx = self.find(title)
        if idx is None:
            return None, pd.DataFrame()
        sims = linear_kernel(self.X[idx], self.X).ravel()
        yrs = self.df["year"].to_numpy(dtype=float)
        gap = np.abs(yrs - yrs[idx])
        sims = sims + np.where(np.isnan(gap), 0.0, 0.05 * np.clip(1 - gap / 30, 0, 1))
        sims[idx] = -1
        order = np.argsort(-sims)[: n * 5]
        out = self.df.iloc[order].copy()
        out["score"] = sims[order]
        out = out[out["title"] != self.df.at[idx, "title"]]
        if min_year:
            out = out[out["year"] >= min_year]
        return self.df.at[idx, "title"], out.head(n)[["title", "genres", "score"]].round(3)


if __name__ == "__main__":
    df = load_and_clean(sys.argv[1] if len(sys.argv) > 1 else "movies.csv")
    eda(df)
    rec = Recommender(df)
    for q in ["Toy Story (1995)", "Matrix, The (1999)", "Titanic (1997)", "Alien (1979)", "Pulp Fiction (1994)"]:
        matched, res = rec.recommend(q, n=5)
        print(f"\n=== Because you liked: {matched} ===")
        print(res.to_string(index=False))