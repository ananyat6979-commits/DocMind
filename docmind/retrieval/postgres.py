"""Persistent pgvector search; inserts and deletes update the index transactionally."""
import json
import numpy as np
from docmind.models import SearchResult
from docmind.embeddings import compatible_embedding_identity
from docmind.postgres import DatabaseScope
from docmind.db import connect
from docmind.store import _row_to_chunk,EmbeddingModelMismatch


class PostgresDenseRetriever:
    def __init__(self,store):
        if not isinstance(store.db_path,DatabaseScope):raise ValueError("Explicit database tenant required")
        self.store=store
    def search(self,query,top_k=5,file_ids=None):
        from docmind.embeddings import embed_query
        return self.search_vector(embed_query(query)[0],top_k,file_ids)
    def search_vector(self,vector,top_k=5,file_ids=None,exact=False):
        vector=np.asarray(vector,dtype='float32')
        if vector.shape!=(384,) or not np.isfinite(vector).all():raise ValueError("Expected finite 384-dimensional vector")
        if not 1<=int(top_k)<=1000:raise ValueError("Search count outside bounds")
        if abs(float(np.linalg.norm(vector))-1)>.002:raise ValueError("Normalized query vector required")
        encoded=json.dumps(vector.tolist())
        with connect(self.store.db_path) as conn:
            conn.corpus_lock(shared=True)
            signature=self.store._get_meta(conn,'embedding_signature','')
            if not compatible_embedding_identity(signature,self.store._embedding_signature()):raise EmbeddingModelMismatch("Active embedding identity differs")
            conn.execute("SET LOCAL hnsw.ef_search=128")
            conn.execute("SET LOCAL hnsw.iterative_scan='strict_order'")
            params=[encoded,self.store.db_path.tenant]
            where="tenant_id=?"
            if file_ids is not None:
                ids=list(file_ids)
                if not ids:return []
                if len(ids)>1000:raise ValueError("Too many file filters")
                where+=" AND file_id IN ("+','.join('?' for _ in ids)+")";params.extend(ids)
            params.extend([encoded,int(top_k)])
            if exact or file_ids is not None:
                statement=f"WITH eligible AS MATERIALIZED (SELECT * FROM chunks WHERE {where}) SELECT *, -(embedding_vector <#> ?::vector) AS similarity FROM eligible ORDER BY embedding_vector <#> ?::vector LIMIT ?"
                params=[self.store.db_path.tenant]+(ids if file_ids is not None else [])+[encoded,encoded,int(top_k)]
            else:
                statement=f"SELECT *, -(embedding_vector <#> ?::vector) AS similarity FROM chunks WHERE {where} ORDER BY embedding_vector <#> ?::vector LIMIT ?"
            rows=conn.execute(statement,params).fetchall()
        return [SearchResult(chunk=_row_to_chunk(row),score=float(row['similarity']),retriever='dense') for row in rows]
