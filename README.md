Movie Recommender

A content-based movie recommender built on the MovieLens dataset. It cleans and explores the metadata (titles, release years, genres), converts genres and release decade into TF-IDF features, and ranks similar films by cosine similarity with a year-proximity tiebreaker. Includes EDA plots, sample queries, and a small Flask API endpoint.

Dataset

movies.csv from MovieLens (62,423 movies) with three columns: movieId, title, genres.

Download it from the MovieLens site and place it in the project folder.

Cleaning and EDA
Extracts the release year from the title (e.g. Toy Story (1995)); 412 titles have no year.
Treats (no genres listed) (5,062 movies) as empty, not as a genre.
Splits pipe-separated genres into lists.
Removes exact duplicate rows; 98 titles appear twice with different movieIds.
python recommender.py saves eda.png with top genres, movies per release year, and genres per movie.
How it works
Each movie becomes a short text: its genres plus a decade token (e.g. Adventure Animation Children decade_1990).
TF-IDF turns that text into vectors.
Cosine similarity between the query movie and every other movie gives a score.
Genres alone create many exact ties, so a small bonus (up to 0.05) is added for movies released close to the query movie's year.
The query movie and same-title duplicates are removed from the results.

Title lookup tries an exact match, then a substring match, then a fuzzy match, so toy story works.

Setup
bash
pip install -r requirements.txt
Usage

Run EDA and sample queries:

bash
python recommender.py movies.csv

Use it in Python:

python
from recommender import Recommender, load_and_clean

rec = Recommender(load_and_clean("movies.csv"))
matched, results = rec.recommend("Titanic", n=5)
print(results)

Run the Flask API:

bash
python app.py
GET /recommend?title=Toy Story&n=5

Returns JSON with the matched title and a list of recommendations (title, genres, score). Returns 400 if title is missing and 404 if no movie matches.

Sample results
Because you liked	Recommendations
Toy Story (1995)	Aladdin (1992), Antz (1998), Toy Story 2 (1999), DuckTales: The Movie (1990)
Matrix, The (1999)	eXistenZ (1999), The Time Shifters (1999), Interceptors (1999)
Titanic (1997)	The Wings of the Dove (1997), Mrs. Dalloway (1997), Jane Eyre (1997)
Pulp Fiction (1994)	Freeway (1996), Man Bites Dog (1992), Fargo (1996)
Limitations
Only genres and release decade are used, so results cluster by genre and many movies tie.
There is no popularity or quality signal, so obscure films can rank next to famous ones.
Possible improvements: add MovieLens tags.csv or genome data for richer text features, and add ratings.csv for popularity filtering or collaborative filtering.
Project structure
recommender.py    cleaning, EDA, and recommender
app.py            Flask endpoint
requirements.txt  dependencies
eda.png           EDA plots
