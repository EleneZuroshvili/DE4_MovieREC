"""
Script to load movies_metadata.csv into ChromaDB for RAG-based movie recommendations.
Reads the TMDB CSV (with JSON-like genre/company columns) and creates searchable documents.

Usage:
    python chatbot/create_movie_db.py
"""

import ast
import chromadb
import pandas as pd
from pathlib import Path


def parse_names(raw_value) -> str:
    """
    Parse stringified list-of-dicts like "[{'id': 16, 'name': 'Animation'}, ...]"
    and return a comma-separated string of names.
    """
    if pd.isna(raw_value) or str(raw_value).strip() in ("", "nan", "[]"):
        return "Unknown"
    try:
        items = ast.literal_eval(str(raw_value))
        if isinstance(items, list):
            return ", ".join(item["name"] for item in items if "name" in item)
    except (ValueError, SyntaxError):
        pass
    return str(raw_value)


def safe_float(val, default=0.0):
    try:
        return float(val) if pd.notna(val) else default
    except (ValueError, TypeError):
        return default


def safe_int(val, default=0):
    try:
        return int(float(val)) if pd.notna(val) else default
    except (ValueError, TypeError):
        return default


def prepare_movie_documents(csv_path: str) -> dict:
    """
    Convert movies_metadata CSV into ChromaDB-ready documents.
    Filters to released movies with an overview and reasonable data quality.
    """
    df = pd.read_csv(csv_path, low_memory=False)

    # --- Data quality filters ---
    # Keep only released movies with an overview
    df = df[df["status"] == "Released"].copy()
    df = df[df["overview"].notna() & (df["overview"].str.len() > 10)]
    # Drop rows where title is missing
    df = df[df["title"].notna()]
    # Keep movies with at least some votes (filters obscure entries)
    df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0)
    df = df[df["vote_count"] >= 10]
    # Reset index
    df = df.reset_index(drop=True)

    print(f"  After filtering: {len(df)} movies (from original CSV)")

    documents = []
    metadatas = []
    ids = []

    for index, row in df.iterrows():
        title = str(row.get("title", "Unknown")).strip()
        overview = str(row.get("overview", "")).strip()
        genres = parse_names(row.get("genres"))
        release_date = str(row.get("release_date", ""))
        vote_average = safe_float(row.get("vote_average"))
        vote_count = safe_int(row.get("vote_count"))
        popularity = safe_float(row.get("popularity"))
        runtime = safe_int(row.get("runtime"))
        tagline = str(row.get("tagline", "")).strip()
        original_language = str(row.get("original_language", "en")).strip()
        production_companies = parse_names(row.get("production_companies"))
        adult = str(row.get("adult", "False")).strip()

        # Clean up nan strings
        if tagline == "nan":
            tagline = ""
        if original_language == "nan":
            original_language = "en"

        # Extract year from release_date
        year = "Unknown"
        if release_date and release_date not in ("Unknown", "nan", ""):
            try:
                year = str(release_date)[:4]
            except (ValueError, IndexError):
                year = "Unknown"

        # Create rich document text optimized for semantic search
        doc_parts = [
            f"Movie: {title}",
            f"Year: {year}",
            f"Genres: {genres}",
            f"Overview: {overview}",
        ]
        if tagline:
            doc_parts.append(f"Tagline: {tagline}")
        doc_parts.append(f"Rating: {vote_average}/10 ({vote_count} votes)")
        if runtime > 0:
            doc_parts.append(f"Runtime: {runtime} minutes")
        doc_parts.append(f"Language: {original_language}")
        if production_companies != "Unknown":
            doc_parts.append(f"Production: {production_companies}")
        doc_parts.append("")
        doc_parts.append(
            f'This is a {genres.lower()} movie titled "{title}" released in {year}. {overview}'
        )

        document_text = "\n".join(doc_parts)

        # Metadata for filtering and display (ChromaDB metadata values must be str/int/float)
        metadata = {
            "title": title.lower(),
            "year": year,
            "genres": genres.lower(),
            "vote_average": vote_average,
            "vote_count": vote_count,
            "popularity": popularity,
            "runtime": runtime,
            "original_language": original_language,
            "adult": adult.lower(),
            "tagline": tagline,
        }

        documents.append(document_text)
        metadatas.append(metadata)
        ids.append(f"movie_{index}")

    return {"documents": documents, "metadatas": metadatas, "ids": ids}


def setup_movie_chromadb(csv_path: str, collection_name: str = "movie_db"):
    """
    Create and populate ChromaDB collection with movie data.
    """
    chroma_path = Path(__file__).parent.parent / "chroma"
    client = chromadb.PersistentClient(path=str(chroma_path))

    # Delete collection if it already exists, then recreate
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={
            "description": "Movie database for recommendation system"
        },
    )

    # Prepare documents from CSV
    data = prepare_movie_documents(csv_path)

    # ChromaDB add() has a batch limit, so add in chunks
    batch_size = 500
    total = len(data["documents"])
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        collection.add(
            documents=data["documents"][start:end],
            metadatas=data["metadatas"][start:end],
            ids=data["ids"][start:end],
        )
        print(f"  Added batch {start}-{end} of {total}")

    print(f"\nSuccessfully added {total} movies to ChromaDB collection '{collection_name}'")
    return collection


if __name__ == "__main__":
    csv_path = Path(__file__).parent.parent / "data" / "movies_metadata.csv"

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Place your movies CSV there first.")
        exit(1)

    print(f"Loading movies from: {csv_path}")
    collection = setup_movie_chromadb(str(csv_path))

    # Quick verification
    print("\n--- Quick test: searching for 'action adventure' ---")
    results = collection.query(query_texts=["action adventure"], n_results=3)
    for i, doc in enumerate(results["metadatas"][0]):
        print(f"  {doc['title'].title()} ({doc['year']}) - {doc['genres']} - {doc['vote_average']}/10")
