from langchain_voyageai import VoyageAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from pymongo import MongoClient
import sys
from flask import Flask, Response, stream_with_context
import time

import json

from flask import Flask, render_template, request, redirect, url_for, jsonify
import os

app = Flask(__name__)

config_store = {
    "openai-key": None,
    "voyageai-key": "",
    "selectedEmbeddingModel": None,
    "mongodb_uri": None,
    "mongodb_database": None,
    "mongodb_collection": None
}



@app.route("/")
def main():
    return render_template("main.html")

@app.route("/config")
def config():
    return render_template("config.html")

database_client = []
# New endpoint for saving config
@app.route("/config/save", methods=["POST"])
def config_save():
    global database_client
    # Retrieve form data
    openai_key = request.form.get("openai_key")
    embedding_model = request.form.get("embedding_model")
    voyageai_key = request.form.get("voyageai_key") if embedding_model == "voyageai" else None

    # TODO: persist these settings (e.g., write to file or database)
    print(f"OpenAI Key: {openai_key}")
    config_store["openai-key"]=openai_key
    config_store["selectedEmbeddingModel"]=embedding_model

    if voyageai_key:
        print(f"VoyageAI Key: {voyageai_key}")
        config_store["voyageai-key"]=voyageai_key

    config_store["mongodb_uri"] = request.form.get("mongo_uri")
    config_store["mongodb_database"] = request.form.get("db_name")
    config_store["mongodb_collection"]  = request.form.get("collection_name")


    client = MongoClient(config_store["mongodb_uri"], appname="Chat with your PDF with Atlas Search")
    try:
        client.admin.command('ping')
        database_client = client
        # Persist configuration to config.json
        with open('config.json', 'w') as f:
            json.dump(config_store, f, indent=2)

        return jsonify({"success":"MongoDB connection successful"})
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        return jsonify({"error": "MongoDB connection failed", "details": str(e)}), 500

   


    # Redirect back to the config page
    #return redirect(url_for("config"))

from flask import jsonify

@app.route('/message')
def get_message():
    # Replace this with whatever dynamic logic you need
    # Load persisted configuration from config.json on startup
    try:
        with open('config.json', 'r') as f:
            loaded = json.load(f)
            config_store.update(loaded)
    except FileNotFoundError:
        pass


    print(f"This is the message: {config_store}")

    return jsonify(config_store)

def getEmbeddingModelAndDimension():
    embedding_models = {
        "openai" : {
            "name" : "Open-AI-Embeddings",
            "dimensions": 1536
        },
        "mpnet" : {
            "name": "MPNet-Base-V2",
            "dimensions" : 768
        },
        "voyageai": {
            "name": "Voyage-AI-3.5-lite",
            "dimensions": 1024
        }
    }

    embedding_model = None
    if config_store["selectedEmbeddingModel"] == "openai":
        return {"dimensions": embedding_models["openai"]["dimensions"], "model": OpenAIEmbeddings(openai_api_key=config_store["openai-key"])}
    elif config_store["selectedEmbeddingModel"] == "voyageai":
        return {"dimensions": embedding_models["voyageai"]["dimensions"], "model": VoyageAIEmbeddings(voyage_api_key=config_store["voyageai-key"],model="voyage-3.5-lite")}
    elif config_store["selectedEmbeddingModel"] == "mpnet":
        return {"dimensions": embedding_models["mpnet"]["dimensions"], "model": HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")}


@app.route('/empty-repo')
def emptyRepo():
    client = database_client
    db = client[config_store["mongodb_database"]]
    collection = db[config_store["mongodb_collection"]]
    collection.drop()
    return jsonify({"message":"Repostitory emptied"})
 

vector_store = None
@app.route('/create-index')
def createVectorIndex():
    client = database_client
    db = client[config_store["mongodb_database"]]
    collection = db[config_store["mongodb_collection"]]
    vector_store = MongoDBAtlasVectorSearch(
        collection,
        getEmbeddingModelAndDimension()["model"],
        index_name="vector_index",
        text_key="text",
        embedding_key="embedding",
    )
    vector_store.create_vector_search_index(getEmbeddingModelAndDimension()["dimensions"])
    return jsonify({"message":"Vector Search Index has been created"})

@app.route("/build-repo", methods=["GET"])
def buildRepo():

    folder = request.args.get("folder")   # None if not provided
    print(f"This is the message coming: {folder}")

    client = database_client
    db = client[config_store["mongodb_database"]]
    collection = db[config_store["mongodb_collection"]]
    vector_store = MongoDBAtlasVectorSearch(
        collection,
        getEmbeddingModelAndDimension()["model"],
        index_name="vector_index",
        text_key="text",
        embedding_key="embedding",
    )
    folder_path = "document-repository/" + folder
    files = [] 

    for filename in os.listdir(folder_path):
        f = os.path.join(folder_path, filename)
        files.append(f)
    

    print("Ready for event stream")
    def event_stream():
        print("Event stream function")
        # In a real buildRepo you’d hook into your long-running process
        '''for i in range(1, 11):
            yield f"data: Step {i} completed\n\n"
            time.sleep(1)
        yie ld "data: Done!\n\n"
        '''

        print("Files:"+str(len(files)))
        for file in files:
            #print(f"The file is being tokenized: {file}")
            print( f"The file is being tokenized: {file} \n\n")
            yield f"data: The file is being tokenized: {file} \n\n"
            loader = PyPDFLoader(file)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)
            vector_store.add_documents(texts)
            #print(f"The file has been tokenized and vectorized and added to vector store: {file}")
            yield f"data: The file has been tokenized and vectorized and added to vector store: {file} \n\n"
        yield f"data: All completed \n\n"

    return Response(
        stream_with_context(event_stream()),
        mimetype='text/event-stream'
    )
    #return jsonify({"files_vectorized": num_of_files})



@app.route('/repositories')
def repositories():
    # Path to the document-repository folder
    repo_root = os.path.join(os.getcwd(), 'document-repository')
    try:
        entries = os.listdir(repo_root)
        # Filter to include only directories
        folders = [name for name in entries if os.path.isdir(os.path.join(repo_root, name))]
    except Exception as e:
        print(f"Error listing repositories: {e}")
        folders = []
    return jsonify({"repositories": folders})


@app.route("/build")
def build():
    #build_up_repository()
    return render_template("build.html", message="Repository built successfully.")


def process_query(query):

    # Set up retriever and language model
    client = database_client
    db = client[config_store["mongodb_database"]]
    collection = db[config_store["mongodb_collection"]]
    vector_store = MongoDBAtlasVectorSearch(
        collection,
        getEmbeddingModelAndDimension()["model"],
        index_name="vector_index",
        text_key="text",
        embedding_key="embedding",
    )
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=config_store["openai-key"])

    # Set up RAG pipeline
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    result = qa_chain.invoke({"query": query})
    return result["result"], result["source_documents"]

@app.route("/ask", methods=["GET", "POST"])
def ask():
    answer = None
    sources = None
    q = None
    if request.method == "POST":
        q = request.form.get("query")
        answer, sources = process_query(q)
    return render_template("ask.html", question=q, answer=answer, sources=sources)

if __name__ == "__main__":
    app.run(debug=True)