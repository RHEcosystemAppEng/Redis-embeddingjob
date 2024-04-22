#!/usr/bin/env python
# coding: utf-8

# ## Creating an index and populating it with documents using Redis
# 
# Simple example on how to ingest PDF documents, then web pages content into a Redis VectorStore.
# 
# Requirements:
# - A Redis cluster
# - A Redis database with at least 2GB of memory (to match with the initial index cap)

# ### Base parameters, the Redis info


from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.redis import Redis
import os
from vector_db.db_provider_factory import DBFactory

# In[17]:

type = os.getenv('DB_TYPE') if os.getenv('DB_TYPE') else "REDIS"
db_provider = DBFactory().create_db_provider(type)

# #### Imports

# In[18]:

# ## Initial index creation and document ingestion

# #### Document loading from a folder containing PDFs

# In[19]:

pdf_folder_path = 'rhods-doc'
loader = PyPDFDirectoryLoader(pdf_folder_path)
docs = loader.load()

# #### Split documents into chunks with some overlap

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024,
                                               chunk_overlap=40)
all_splits = text_splitter.split_documents(docs)


# #### Create the index and ingest the documents

# In[21]:

db_provider.add_documents(all_splits)

# ## Ingesting new documents
# #### Example with Web pages

# In[23]:

from langchain.document_loaders import WebBaseLoader

# In[24]:

loader = WebBaseLoader(["https://ai-on-openshift.io/getting-started/openshift/",
                        "https://ai-on-openshift.io/getting-started/opendatahub/",
                        "https://ai-on-openshift.io/getting-started/openshift-ai/",
                        "https://ai-on-openshift.io/odh-rhoai/configuration/",
                        "https://ai-on-openshift.io/odh-rhoai/custom-notebooks/",
                        "https://ai-on-openshift.io/odh-rhoai/nvidia-gpus/",
                        "https://ai-on-openshift.io/odh-rhoai/custom-runtime-triton/",
                        "https://ai-on-openshift.io/odh-rhoai/openshift-group-management/",
                        "https://ai-on-openshift.io/tools-and-applications/minio/minio/",
                        "https://access.redhat.com/articles/7047935",
                        "https://access.redhat.com/articles/rhoai-supported-configs",
                       ])

# In[25]:
data = loader.load()

# In[26]:
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024,
                                               chunk_overlap=40)
all_splits = text_splitter.split_documents(data)

# In[27]:
db_provider.add_documents(all_splits)