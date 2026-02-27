from enum import StrEnum

import psycopg2
from psycopg2.extras import RealDictCursor

from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.utils.text import chunk_text


class SearchMode(StrEnum):
    EUCLIDIAN_DISTANCE = "euclidean"  # Euclidean distance (<->)
    COSINE_DISTANCE = "cosine"  # Cosine distance (<=>)

_EMBEDDINGS_TABLE_NAME = "vectors"

class TextProcessor:
    """Processor for text documents that handles chunking, embedding, storing, and retrieval"""

    def __init__(self, embeddings_client: DialEmbeddingsClient, db_config: dict):
        self.embeddings_client = embeddings_client
        self.db_config = db_config

    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(
            host=self.db_config['host'],
            port=self.db_config['port'],
            database=self.db_config['database'],
            user=self.db_config['user'],
            password=self.db_config['password']
        )

    # provide method `process_text_file` that will:
    #   - apply file name, chunk size, overlap, dimensions and bool of the table should be truncated
    #   - truncate table with vectors if needed
    #   - load content from file and generate chunks (in `utils.text` present `chunk_text` that will help do that)
    #   - generate embeddings from chunks
    #   - save (insert) embeddings and chunks to DB
    #       hint 1: embeddings should be saved as string list
    #       hint 2: embeddings string list should be casted to vector ({embeddings}::vector)
    def process_text_file(self, file_name: str, chunk_size: int, overlap: int, dimensions: int, truncate_table: bool = False):
        if truncate_table:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    print(f"==> Truncating {_EMBEDDINGS_TABLE_NAME} table...", end='')
                    cursor.execute(f"TRUNCATE TABLE {_EMBEDDINGS_TABLE_NAME}")
                    conn.commit()
                    print("Done!")

        print(f"==> Processing file `{file_name}`...", end='')
        with open(file_name, 'r') as file:
            text = file.read()
        print("Done! text length:", len(text), "\n==> Generating chunks and embeddings...", end='')

        chunks = chunk_text(text, chunk_size, overlap)
        print(f"Done! {len(chunks)} chunks generated!\n==> Generating embeddings...", end='')
        embeddings_dict = self.embeddings_client.get_embeddings(chunks)
        print(f"Done! {len(embeddings_dict.keys())} embeddings generated!\n==> Saving to DB...", end='')

        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                for index, embedding in embeddings_dict.items():
                    chunk = chunks[index]
                    embedding_str = str(embedding)
                    document_name = file_name.split('/')[-1]
                    cursor.execute(
                        f"INSERT INTO {_EMBEDDINGS_TABLE_NAME} (document_name, text, embedding) VALUES (%s, %s, %s::vector)",
                        (document_name, chunk, embedding_str)
                    )
                conn.commit()

        print("Done!")

    #TODO:
    # provide method `search` that will:
    #   - apply search mode, user request, top k for search, min score threshold and dimensions
    #   - generate embeddings from user request
    #   - search in DB relevant context
    #     hint 1: to search it in DB you need to create just regular select query
    #     hint 2: Euclidean distance `<->`, Cosine distance `<=>`
    #     hint 3: You need to extract `text` from `vectors` table
    #     hint 4: You need to filter distance in WHERE clause
    #     hint 5: To get top k use `limit`

