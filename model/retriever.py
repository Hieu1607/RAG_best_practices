from faiss import IDSelectorArray, SearchParameters
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch

import spacy
import faiss

# Load the English model
nlp = spacy.load("en_core_web_sm")

class Retriever:
    """
    Handles the retrieval of relevant documents from a pre-built FAISS index.
    Enables querying with sentence transformers embeddings.

    Attributes:
        index (faiss.Index): FAISS index for fast similarity search.
        doc_info (pd.DataFrame): DataFrame containing detailed information about documents.
        documents (list of str): List of original documents.
        embedding_model (SentenceTransformer): Model used for embedding the documents and queries.
    """

    def __init__(self, index, doc_info, embedding_model_name, model_loader_seq2seq, index_titles, index_icl=None, icl_info=None):
        """Initializes the Retriever with all necessary components for document retrieval.

        Sets up the embedding model, FAISS indices, and seq2seq model for query expansion.
        Prepares the retriever to find relevant documents based on semantic similarity.

        Args:
            index (faiss.Index): Pre-built FAISS index containing document embeddings for similarity search.
            doc_info (pd.DataFrame): Metadata about documents (text, IDs, etc.) aligned with index.
            embedding_model_name (str): Name of the SentenceTransformer model (e.g., 'all-MiniLM-L6-v2').
            model_loader_seq2seq (ModelLoader): Loader containing seq2seq model for query expansion.
            index_titles (faiss.Index): FAISS index of document titles for filtering during query expansion.
            index_icl (faiss.Index, optional): FAISS index for ICL examples (hybrid mode only).
            icl_info (pd.DataFrame, optional): Metadata about ICL examples (hybrid mode only).
        """
        self.index = index
        self.doc_info = doc_info
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model = SentenceTransformer(embedding_model_name).to(self.device)
        self.sent_info = None
        self.index_sents = None
        
        # Hybrid mode support
        self.index_icl = index_icl
        self.icl_info = icl_info
        self.hybrid_mode = (index_icl is not None and icl_info is not None)

        self.model_seq2seq = model_loader_seq2seq.model
        self.tokenizer_seq2seq = model_loader_seq2seq.tokenizer
        # Define text-query pairs for query expansion
        self.text_query_pairs = [
            {"text": "Mitochondria play a crucial role in cellular respiration and energy production within human cells.", "query": "Cell Biology, Mitochondria, Energy Metabolism"},
            {"text": "The Treaty of Versailles had significant repercussions that contributed to the onset of World War II.", "query": "World History, Treaty of Versailles, World War II"},
            {"text": "What are the implications of the Higgs boson discovery for particle physics and the Standard Model?", "query": "Particle Physics, Higgs Boson, Standard Model"},
            {"text": "How did the Silk Road influence cultural and economic interactions during the Middle Ages?", "query": "Silk Road, Middle Ages, Cultural Exchange"}
        ]
        self.index_titles = index_titles

    def build_index(self, documents):
        """Builds a FAISS index from documents for fast similarity-based retrieval.

        This method splits documents into sentences, generates embeddings for each sentence,
        and creates a FAISS index optimized for cosine similarity search (Inner Product).

        Args:
            documents (list of str): List of text documents to be indexed.

        Returns:
            faiss.IndexFlatIP: FAISS index containing sentence embeddings ready for search.
        """
        embeddings = self.embed_sents(documents)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        return index

    def embed_sents(self, documents):
        """Generates vector embeddings for all sentences in the given documents.

        Processing pipeline:
        1. Splits documents into individual sentences using prepare_sents()
        2. Encodes each sentence into a dense vector using the SentenceTransformer model
        3. Stores sentence metadata and embeddings in self.sent_info DataFrame

        Args:
            documents (list of str): List of text documents to process.

        Returns:
            np.ndarray: Array of embeddings with shape [num_sentences, embedding_dim].
        """
        self.sent_info = self.prepare_sents(documents)
        self.sent_info = pd.DataFrame(self.sent_info)
        embeddings = self.embedding_model.encode(self.sent_info["text"].tolist(), show_progress_bar=True)
        self.sent_info['embedding'] = embeddings.tolist()

        return np.array(embeddings)
    
    def prepare_sents(self, documents):
        """Splits documents into individual sentences and creates metadata for each sentence.

        Uses Spacy's sentence tokenizer to parse documents into sentences, then assigns
        a unique ID to each sentence for tracking purposes.

        Args:
            documents (list of str): List of text documents to split.

        Returns:
            list of dict: List containing metadata for each sentence:
                - 'text': The sentence text
                - 'org_sent_id': Unique sentence ID for reference
        """
        sent_info = []
        sent_id = 0
        for document in documents:
            
            doc = nlp(document)
            sents = [sent.text for sent in doc.sents]
            
            # Prepend same document to its chunks and store document/chunk details
            for sent in sents:
                sent_dict = {"text": sent, "org_sent_id": sent_id}
                sent_info.append(sent_dict)
                sent_id += 1
        return sent_info

    def retrieve(self, query_batch, k, expand_query, k_titles, icl_kb_idx_batch=None, focus=None, k_icl=0):
        """Retrieves the top-k most relevant documents for each query using semantic similarity.

        Main retrieval pipeline:
        1. (Optional) Expands queries into keywords using seq2seq model and filters by titles
        2. Encodes queries into embeddings
        3. Searches FAISS index for most similar documents
        4. (Optional) In hybrid mode, also retrieves ICL examples from separate index
        5. (Optional) Refines results by finding most relevant sentences within documents
        6. Formats and returns results with similarity scores

        Args:
            query_batch (list of str): Batch of query strings to process.
            k (int): Number of documents to retrieve per query.
            expand_query (bool): Whether to expand queries into keywords for better retrieval.
            k_titles (int): Number of titles to search when expanding queries.
            icl_kb_idx_batch (list of int, optional): Indices to exclude (e.g., correct answers in ICL).
            focus (int, optional): If set, refines search to top N sentences within retrieved docs.
            k_icl (int, optional): Number of ICL examples to retrieve (hybrid mode only).

        Returns:
            list of list of dict: Nested list where each inner list contains retrieved results
                for the corresponding query. Each result dict includes:
                - 'text': Document/sentence text
                - 'doc_id' or 'sent_id': Document or sentence identifier
                - 'score': Similarity score
                - 'correct_answer', 'incorrect_answer': (Only if icl_kb_idx_batch provided or hybrid mode)
                - 'source': 'icl' or 'article' (Only in hybrid mode)
        """

        if k == 0 and k_icl == 0:
            return [[] for _ in query_batch]
        
        # Retrieve ICL examples if hybrid mode
        icl_results_batch = []
        if self.hybrid_mode and k_icl > 0:
            icl_results_batch = self._retrieve_icl(query_batch, k_icl, icl_kb_idx_batch)
        
        # Retrieve articles (original logic)
        article_results_batch = self._retrieve_articles(query_batch, k, expand_query, k_titles, icl_kb_idx_batch, focus)
        
        # Combine results: ICL first, then articles
        if self.hybrid_mode and k_icl > 0:
            combined_results = []
            for icl_results, article_results in zip(icl_results_batch, article_results_batch):
                combined_results.append(icl_results + article_results)
            return combined_results
        else:
            return article_results_batch
    
    def _retrieve_icl(self, query_batch, k_icl, icl_kb_idx_batch):
        """Retrieves ICL examples from the ICL index.
        
        Args:
            query_batch (list of str): Batch of query strings.
            k_icl (int): Number of ICL examples to retrieve.
            icl_kb_idx_batch (list of int, optional): Indices to exclude.
            
        Returns:
            list of list of dict: ICL examples for each query.
        """
        query_embeddings = self.embedding_model.encode(query_batch, show_progress_bar=False)
        
        results_batch = []
        for i, query_embedding in enumerate(query_embeddings):
            # Filter out current query if needed
            if icl_kb_idx_batch:
                all_ids = list(range(self.index_icl.ntotal))
                all_ids.remove(icl_kb_idx_batch[i])
                id_selector = IDSelectorArray(all_ids)
                similarities, indices = self.index_icl.search(
                    np.array([query_embedding]), k_icl, 
                    params=SearchParameters(sel=id_selector)
                )
            else:
                similarities, indices = self.index_icl.search(np.array([query_embedding]), k_icl)
            
            indices, similarities = indices[0], similarities[0]

            # Sanitize indices: remove -1 and out-of-bounds
            total_icl = len(self.icl_info) if self.icl_info is not None else 0
            valid_indices = [int(idx) for idx in indices if 0 <= int(idx) < total_icl]
            valid_sims = [sim for idx, sim in zip(indices, similarities) if 0 <= int(idx) < total_icl]

            # Create results with 'source' tag
            results = []
            for idx, sim in zip(valid_indices, valid_sims):
                icl_item = self.icl_info.iloc[idx]
                result = {
                    "text": icl_item["text"],
                    "doc_id": icl_item["org_doc_id"],
                    "score": sim,
                    "correct_answer": icl_item["correct_answer"],
                    "incorrect_answer": icl_item["incorrect_answer"],
                    "source": "icl"
                }
                results.append(result)
            results_batch.append(results)
        
        return results_batch
    
    def _retrieve_articles(self, query_batch, k, expand_query, k_titles, icl_kb_idx_batch, focus):
        """Retrieves articles from the main document index.
        
        This is the original retrieve logic for articles/documents.
        """
        if k == 0:
            return [[] for _ in query_batch]

        if expand_query:
            # Expand the query using a seq2seq model
            eq_prompt_batch_str = []
            for query in query_batch:
                examples = self.text_query_pairs.copy()
                examples.append({"text": query, "query": ""})
                eq_prompt = "\n".join([f"Question: {example['text']}\nQuery Keywords: {example['query']}" for example in examples])
                eq_prompt_batch_str.append(eq_prompt)

            eq_prompt_batch_enc = self.tokenizer_seq2seq(eq_prompt_batch_str, return_tensors='pt', padding=True).to(self.device)
            eq_batch_enc = self.model_seq2seq.generate(**eq_prompt_batch_enc, max_length=25, num_return_sequences=1)
            eq_batch = self.tokenizer_seq2seq.batch_decode(eq_batch_enc, skip_special_tokens=True)
            eq_batch = [eq.split(", ") for eq in eq_batch] # Split the expanded queries

            # Encode the expanded queries and search the index for similar titles
            eq_batch_indexed = [(eq, i) for i, eqs in enumerate(eq_batch) for eq in eqs]
            eq_batch_flat = [eq for eq, _ in eq_batch_indexed]
            eq_embeddings = self.embedding_model.encode(eq_batch_flat, show_progress_bar=False)
            _, indices_eq = self.index_titles.search(np.array(eq_embeddings), k_titles)

            # Retrieve the indices of the documents associated with the similar titles
            indices_eq_batch = [[] for _ in range(len(query_batch))]
            for ids, (_, i) in zip(indices_eq, eq_batch_indexed):
                indices_eq_batch[i].append(self.doc_info[self.doc_info['org_doc_id'].isin(ids)].index.tolist())
        else:
            # If not expanding the query, set the indices to an empty list
            if icl_kb_idx_batch:
                # Remove the correct answer from the retrieved documents
                all_ids_batch = [list(range(self.index.ntotal)) for _ in range(len(query_batch))]
                for all_ids, icl_kb_idx in zip(all_ids_batch, icl_kb_idx_batch):
                    all_ids.remove(icl_kb_idx)
                all_ids_batch = [[all_ids] for all_ids in all_ids_batch]
                indices_eq_batch = all_ids_batch
            else:
                indices_eq_batch = [[] for _ in range(len(query_batch))]

        # Batch encode the queries
        query_embeddings = self.embedding_model.encode(query_batch, show_progress_bar=False)

        # Process each query separately
        results_batch = []
        for query_embedding, ids_filter in zip(query_embeddings, indices_eq_batch):
            ids_filter = ids_filter if ids_filter else [list(range(self.index.ntotal))]

            id_filter_set = set()
            for id_filter in ids_filter:
                id_filter_set.update(id_filter)

            id_filter = list(id_filter_set)
            id_selector = IDSelectorArray(id_filter)
            # Search the index for similar documents, retrieve a larger set of documents
            similarities, indices = self.index.search(np.array([query_embedding]), k, params=SearchParameters(sel=id_selector))
            indices, similarities = indices[0], similarities[0]

            # Sanitize document indices
            total_docs = len(self.doc_info) if self.doc_info is not None else 0
            doc_indices = [int(idx) for idx in indices if 0 <= int(idx) < total_docs]
            doc_sims = [sim for idx, sim in zip(indices, similarities) if 0 <= int(idx) < total_docs]
            
            # Focus on the most relevant sentences from the retrieved documents
            if focus:
                if not doc_indices:
                    results_batch.append([])
                    continue
                docs = self.doc_info.iloc[doc_indices]["text"].tolist()
                self.index_sents = self.build_index(docs)
                similarities, indices = self.index_sents.search(np.array([query_embedding]), focus)
                indices, similarities = indices[0], similarities[0]

                # Sanitize sentence indices
                total_sents = len(self.sent_info) if self.sent_info is not None else 0
                sent_indices = [int(idx) for idx in indices if 0 <= int(idx) < total_sents]
                sent_sims = [sim for idx, sim in zip(indices, similarities) if 0 <= int(idx) < total_sents]

            icl_kb = icl_kb_idx_batch!=None
            if focus:
                # Retrieve the most relevant sentences from the retrieved documents
                results_batch.append([self._create_result(idx, sim, icl_kb, focus) for idx, sim in zip(sent_indices[:focus], sent_sims)])
            else:
                results_batch.append([self._create_result(idx, sim, icl_kb, focus) for idx, sim in zip(doc_indices[:k], doc_sims)])

        return results_batch


    def _create_result(self, idx, score, icl_kb, focus):
        """Creates a formatted result dictionary for a retrieved document or sentence.

        Builds a dictionary containing the retrieved text, its identifier, and similarity score.
        The structure varies based on whether focus mode is enabled (sentence-level) or
        disabled (document-level).

        Args:
            idx (int): Index of the document/sentence in doc_info or sent_info.
            score (float): Similarity score from FAISS search.
            icl_kb (bool): Whether to include correct/incorrect answers (for In-Context Learning).
            focus (bool): Whether retrieving sentences (True) or full documents (False).

        Returns:
            dict: Result dictionary containing:
                - 'text': The retrieved text
                - 'doc_id' or 'sent_id': Identifier
                - 'score': Similarity score
                - 'correct_answer', 'incorrect_answer': (Only if icl_kb=True)
        """
        if focus: 
            # Retrieve the most relevant sentences from the retrieved documents
            sent = self.sent_info.iloc[idx]
            result_dict = {
            "text": sent["text"],
            "sent_id": sent["org_sent_id"],
            "score": score
        }
        else:
            doc = self.doc_info.iloc[idx]
            # Create the result dictionary
            result_dict = {
                "text": doc["text"],
                "doc_id": doc["org_doc_id"],
                "score": score
            }

            if icl_kb:
                # Include the correct and incorrect answers for ICL KB
                result_dict['correct_answer'] = doc["correct_answer"]
                result_dict['incorrect_answer'] = doc["incorrect_answer"]

        return result_dict