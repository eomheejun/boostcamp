from rank_bm25 import BM25Okapi, BM25L, BM25Plus
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

from tqdm.auto import tqdm
import pandas as pd
import pickle
import json
import os
import numpy as np

from datasets import (
    Dataset,
    load_from_disk,
    concatenate_datasets,
)
from konlpy.tag import Mecab

import time
from contextlib import contextmanager

#from elasticsearch import Elasticsearch


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.3f} s')

class SparseRetrieval:
    def __init__(self, tokenize_fn, data_path="./data/", context_path="wikipedia_documents.json"):
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v['text'] for v in wiki.values()])) # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        # Transform by vectorizer
        self.tfidfv = TfidfVectorizer(
            tokenizer=tokenize_fn,
            ngram_range=(1, 2),
            max_features=50000,
        )        

        # should run get_sparse_embedding() or build_faiss() or get_BM25_embedding() first.
        self.p_embedding = None
        self.indexer = None

        self.BM25 = None
        self.tokenizer = tokenize_fn
        
        #self.es = Elasticsearch('localhost:9200')
    
    def get_elastic_search(self):
        self.es.indices.create(index = 'document',
                  body = {
                      'settings':{
                          'analysis':{
                              'analyzer':{
                                  'my_analyzer':{
                                      "type": "custom",
                                      'tokenizer':'nori_tokenizer',
                                      'decompound_mode':'mixed',
                                      'stopwords':'_korean_',
                                      "filter": ["lowercase",
                                                 "my_shingle_f",
                                                 "nori_readingform",
                                                 "nori_number"]
                                  }
                              },
                              'filter':{
                                  'my_shingle_f':{
                                      "type": "shingle"
                                  }
                              }
                          },
                          'similarity':{
                              'my_similarity':{
                                  'type':'BM25',
                              }
                          }
                      },
                      'mappings':{
                          'properties':{
                              'title':{
                                  'type':'text',
                                  'analyzer':'my_analyzer',
                                  'similarity':'my_similarity'
                              },
                              'text':{
                                  'type':'text',
                                  'analyzer':'my_analyzer',
                                  'similarity':'my_similarity'
                              }
                          }
                      }
                  }
                  )
        self.es.indices.get('document')
        df_es = pd.read_csv('/opt/ml/code/data/wiki_for_elastic.csv', index_col=0)
        for num in tqdm(range(len(df_es))):
            self.es.index(index='document', body = {"title" : df['title'][num], "text" : df['text'][num]})

    def get_sparse_embedding(self):
        # Pickle save.
        pickle_name = f"sparse_embedding.bin"
        tfidfv_name = f"tfidv.bin"
        emd_path = os.path.join(self.data_path, pickle_name)
        tfidfv_path = os.path.join(self.data_path, tfidfv_name)
        if os.path.isfile(emd_path) and os.path.isfile(tfidfv_path):
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            with open(tfidfv_path, "rb") as file:
                self.tfidfv = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")
            self.p_embedding = self.tfidfv.fit_transform(self.contexts)
            print(self.p_embedding.shape)
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            with open(tfidfv_path, "wb") as file:
                pickle.dump(self.tfidfv, file)
            print("Embedding pickle saved.")
    
    # BM25 피클 형태로 저장하는 건 구현도전...!
    def get_embedding_BM25(self):
        #tokenized_contexts= [self.tokenizer(i) for i in self.contexts]
        #self.BM25 = BM25Plus(tokenized_contexts)

        pickle_name = f"BM25_embedding.bin"        
        emd_path = os.path.join(self.data_path, pickle_name)
        if os.path.isfile(emd_path):
            with open(emd_path, "rb") as file:
                self.BM25 = pickle.load(file)            
            print("BM25 Embedding pickle load.")
        else:
            print("Build passage BM25_embedding")
            tokenized_contexts= [self.tokenizer(i) for i in self.contexts]
            self.BM25 = BM25Plus(tokenized_contexts)           
            with open(emd_path, "wb") as file:
                pickle.dump(self.BM25, file)
            print("BM25 Embedding pickle saved.")



    def build_faiss(self):
        # FAISS build
        num_clusters = 16
        niter = 5

        # 1. Clustering
        p_emb = self.p_embedding.toarray().astype(np.float32)
        emb_dim = p_emb.shape[-1]
        index_flat = faiss.IndexFlatL2(emb_dim)

        clus = faiss.Clustering(emb_dim, num_clusters)
        clus.verbose = True
        clus.niter = niter
        clus.train(p_emb, index_flat)

        centroids = faiss.vector_float_to_array(clus.centroids)
        centroids = centroids.reshape(num_clusters, emb_dim)

        quantizer = faiss.IndexFlatL2(emb_dim)
        quantizer.add(centroids)

        # 2. SQ8 + IVF indexer (IndexIVFScalarQuantizer)
        self.indexer = faiss.IndexIVFScalarQuantizer(quantizer, quantizer.d, quantizer.ntotal, faiss.METRIC_L2)
        self.indexer.train(p_emb)
        self.indexer.add(p_emb)

    def retrieve(self, query_or_dataset, topk=1):
        assert self.p_embedding is not None, "You must build faiss by self.get_sparse_embedding() before you run self.retrieve()."
        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.contexts[doc_indices[i]])
            return doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)]

        elif isinstance(query_or_dataset, Dataset):
            # make retrieved result as dataframe
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(query_or_dataset['question'], k=1)
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Sparse retrieval: ")):
                # relev_doc_ids = [el for i, el in enumerate(self.ids) if i in doc_indices[idx]]
                tmp = {
                    "question": example["question"],
                    "id": example['id'],
                    "context_id": doc_indices[idx][0],  # retrieved id
                    "context": self.contexts[doc_indices[idx][0]]  # retrieved doument
                }
                if 'context' in example.keys() and 'answers' in example.keys():
                    tmp["original_context"] = example['context']  # original document
                    tmp["answers"] = example['answers']           # original answer
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query, k=1):
        """
        참고: vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """
        with timer("transform"):
            query_vec = self.tfidfv.transform([query])
        assert (
                np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        with timer("query ex search"):
            result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        sorted_result = np.argsort(result.squeeze())[::-1]
        return result.squeeze()[sorted_result].tolist()[:k], sorted_result.tolist()[:k]

    def get_relevant_doc_bulk(self, queries, k=1):
        query_vec = self.tfidfv.transform(queries)
        assert (
                np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices

    # BM25
    def retrieve_BM25(self, query_or_dataset, topk=1):
                
        #assert self.p_embedding is not None, "You must build faiss by self.get_sparse_embedding() before you run self.retrieve()."
        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_BM25(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.contexts[doc_indices[i]])
            return doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)]

        elif isinstance(query_or_dataset, Dataset):
            # make retrieved result as dataframe
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk_BM25(query_or_dataset['question'], k=topk)
            for idx, example in enumerate(tqdm(query_or_dataset, desc="BM25 retrieval: ")):
                topK_context =''
                #topK_scores = ''
                #for i in range(topk):
                for i in range(len(doc_indices[idx])):
                #    topK_context+='ᴥ'
                #    topK_context+=self.contexts[doc_indices[idx][i]]
                #    topK_scores+='ᴥ'
                #    topK_scores+=str(doc_scores[idx][i])                            
                     topK_context+=self.contexts[doc_indices[idx][i]]
                tmp = {
                    "question": example["question"],
                    "id": example['id'],
                    "context_id": doc_indices[idx][0],  # retrieved id, top_0 정보만 담고 있음. 따로 쓰이는데 없어서 그냥 둠 
                    "context": topK_context # retrieved doument, top_k context 이어 붙인 형태로 넘김. 어차피 max_seq_len으로 잘려짐.
                    #"context_score": topK_scores
                }
                if 'context' in example.keys() and 'answers' in example.keys():
                    tmp["original_context"] = example['context']  # original document
                    tmp["answers"] = example['answers']           # original answer
                total.append(tmp)

            cqas = pd.DataFrame(total)

        return cqas
                   
    def get_relevant_doc_BM25(self, query, k=1):
        # 한개씩 돌아가는지 확인 못해봄, inference때 쓰는건 아래코드
        tokenized_query = self.tokenizer(query) 
        
        doc_scores = self.BM25.get_scores(tokenized_query)
        doc_indices = doc_scores.argmax()
        print(doc_scores, doc_indices)
        return doc_scores, doc_indices

    def get_relevant_doc_bulk_BM25(self, queries, k=10):
        # 여기서 시간 좀 걸림.(600개 테스트 셋 5분정도)

        pickle_score_name = f"BM25_relevant_doc_P85_K10_score.bin"     
        pickle_indice_name = f"BM25_relevant_doc_P85_K10_indice.bin"        
        score_path = os.path.join(self.data_path, pickle_score_name)      
        indice_path = os.path.join(self.data_path, pickle_indice_name)
        if os.path.isfile(score_path) and os.path.isfile(indice_path):
            with open(score_path, "rb") as file:
                doc_scores = pickle.load(file)  
            with open(indice_path, "rb") as file:
                doc_indices= pickle.load(file)            
            print("BM25_relevant_doc_P85_K10 pickle load.")
        else:
            print("Build BM25_relevant_doc_P85_K10 pickle")
            tokenized_queries= [self.tokenizer(i) for i in queries]        
            doc_scores = []
            doc_indices = []
            for i in tqdm(tokenized_queries):
                scores = self.BM25.get_scores(i)

                sorted_score = np.sort(scores)[::-1]
                sorted_id = np.argsort(scores)[::-1]
                max_nintypercent = sorted_score>sorted_score[0]*0.85             
            
                if len(sorted_score[max_nintypercent])<=k:
                    doc_scores.append(sorted_score[max_nintypercent])
                    doc_indices.append(sorted_id[max_nintypercent])
                else:
                    # 85퍼센트 score 넘는 passage가 k개 넘으면 자른다. 
                    doc_scores.append(sorted_score[:k])
                    doc_indices.append(sorted_id[:k])
            with open(score_path, "wb") as file:
                pickle.dump(doc_scores, file)
            with open(indice_path, "wb") as file:
                pickle.dump(doc_indices, file)
            print("BM25_relevant_doc_P85_K10 pickle saved.")        

        return doc_scores, doc_indices
    
    # elastic search
    def retrieve_ES(self, query_or_dataset, topk=1):
                
        #assert self.p_embedding is not None, "You must build faiss by self.get_sparse_embedding() before you run self.retrieve()."
        if isinstance(query_or_dataset, str):
            # 작동안함
            doc_scores, doc_indices = self.get_relevant_doc_ES(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.contexts[doc_indices[i]])
            return doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)]

        elif isinstance(query_or_dataset, Dataset):
            # make retrieved result as dataframe
            total = []
            with timer("query exhaustive search"):
                doc = self.get_relevant_doc_bulk_ES(query_or_dataset['question'], k=topk)
            for idx, example in enumerate(tqdm(query_or_dataset, desc="ES retrieval: ")):
                topK_context = ''
                #for i in range(topk):
                for i in range(len(doc[idx])):
                    topK_context+=doc[idx][i]['_source']['text']
                
                tmp = {
                    "question": example["question"],
                    "id": example['id'],
                    "context_id": 0,  # 아 몰라 
                    "context": topK_context # retrieved doument, top_k context 이어 붙인 형태로 넘김. 어차피 max_seq_len으로 잘려짐.
                }
                if 'context' in example.keys() and 'answers' in example.keys():
                    tmp["original_context"] = example['context']  # original document
                    tmp["answers"] = example['answers']           # original answer
                total.append(tmp)

            cqas = pd.DataFrame(total)

        return cqas
                   
    def get_relevant_doc_ES(self, query, k=1):
        # 한개씩 돌아가는지 확인 못해봄, inference때 쓰는건 아래코드
        tokenized_query = self.tokenizer(query) 
        
        doc_scores = self.BM25.get_scores(tokenized_queries)
        doc_indices = doc_scores.argmax()
        print(doc_scores, doc_indices)
        return doc_scores, doc_indices

    def get_relevant_doc_bulk_ES(self, queries, k=1):
        doc = []
        
        for question in queries:
            query = {
                    'query':{
                        'bool':{
                            'must':[
                                    {'match':{'text':question}}
                            ],
                            'should':[
                                    {'match':{'text':question}}
                            ]
                        }
                    }
                }

            documents = self.es.search(index='document',body=query,size=k)['hits']['hits']
            doc.append(documents)
        return doc

    def retrieve_faiss(self, query_or_dataset, topk=1):
        assert self.indexer is not None, "You must build faiss by self.build_faiss() before you run self.retrieve_faiss()."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_faiss(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.contexts[doc_indices[i]])
            return doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)]

        elif isinstance(query_or_dataset, Dataset):
            queries = query_or_dataset['question']
            # make retrieved result as dataframe
            total = []
            with timer("query faiss search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk_faiss(queries, k=topk)
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Sparse retrieval: ")):
                # relev_doc_ids = [el for i, el in enumerate(self.ids) if i in doc_indices[idx]]

                tmp = {
                    "question": example["question"],
                    "id": example['id'],  # original id
                    "context_id": doc_indices[idx][0],  # retrieved id
                    "context": self.contexts[doc_indices[idx][0]]  # retrieved doument
                }
                if 'context' in example.keys() and 'answers' in example.keys():
                    tmp["original_context"]: example['context']  # original document
                    tmp["answers"]: example['answers']           # original answer
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc_faiss(self, query, k=1):
        """
        참고: vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """
        query_vec = self.tfidfv.transform([query])
        assert (
                np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_emb = query_vec.toarray().astype(np.float32)
        with timer("query faiss search"):
            D, I = self.indexer.search(q_emb, k)
        return D.tolist()[0], I.tolist()[0]


    def get_relevant_doc_bulk_faiss(self, queries, k=1):
        query_vecs = self.tfidfv.transform(queries)

        assert (
                np.sum(query_vecs) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."
        q_embs = query_vecs.toarray().astype(np.float32)
        D, I = self.indexer.search(q_embs, k)
        return D.tolist(), I.tolist()


if __name__ == "__main__":
    # Test sparse
    org_dataset = load_from_disk("data/train_dataset")
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    ) # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*"*40, "query dataset", "*"*40)
    print(full_ds)

    ### Mecab 이 가장 높은 성능을 보였기에 mecab 으로 선택 했습니다 ###
    mecab = Mecab()
    def tokenize(text):
        # return text.split(" ")
        return mecab.morphs(text)

    # from transformers import AutoTokenizer
    #
    # tokenizer = AutoTokenizer.from_pretrained(
    #     "bert-base-multilingual-cased",
    #     use_fast=True,
    # )
    ###############################################################

    wiki_path = "wikipedia_documents.json"
    retriever = SparseRetrieval(
        # tokenize_fn=tokenizer.tokenize,
        tokenize_fn=tokenize,
        data_path="data",
        context_path=wiki_path)

    # test single query
    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    with timer("single query by exhaustive search"):
        scores, indices = retriever.retrieve(query)
    with timer("single query by faiss"):
        scores, indices = retriever.retrieve_faiss(query)

    # test bulk
    with timer("bulk query by exhaustive search"):
        df = retriever.retrieve(full_ds)
        df['correct'] = df['original_context'] == df['context']
        print("correct retrieval result by exhaustive search", df['correct'].sum() / len(df))
    with timer("bulk query by exhaustive search"):
        df = retriever.retrieve_faiss(full_ds)
        df['correct'] = df['original_context'] == df['context']
        print("correct retrieval result by faiss", df['correct'].sum() / len(df))


