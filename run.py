import argparse
import shutil
import tempfile
from pathlib import Path
import subprocess

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser


def clone_git_repo(git_url: str, clone_dir: str) -> Path:
    print(f"📥 Cloning repo from: {git_url}")
    repo_path = Path(clone_dir) / Path(git_url).stem
    subprocess.run(["git", "clone", git_url, str(repo_path)], check=True)
    return repo_path


def safe_load_documents(code_path: Path):
    loader = GenericLoader.from_filesystem(
        path=str(code_path),
        glob="**/*.*",
        parser=LanguageParser(),
        show_progress=True,
    )

    all_docs = []
    for blob in loader.blob_loader.yield_blobs():
        try:
            all_docs.extend(loader.blob_parser.lazy_parse(blob))
        except UnicodeDecodeError as e:
            print(f"⚠️ Skipping binary or unreadable file: {blob.path} ({e})")
        except Exception as e:
            print(f"⚠️ Skipping file due to unexpected error: {blob.path} ({e})")
    return all_docs


def build_vectorstore(code_path: Path, persist_dir: str = "./chroma_db"):
    print(f"\n📂 Loading and indexing files from: {code_path}")
    docs = safe_load_documents(code_path)
    print(f"📁 Total files loaded: {len(docs)}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]

    print(f"📄 Total chunks to embed: {len(chunks)}")
    if not chunks:
        raise ValueError("❌ No text chunks to embed. Check file types or project content.")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    vectordb.persist()
    print("✅ Vectorstore built and saved.")
    return vectordb


def start_chat(vectordb):
    retriever = vectordb.as_retriever()
    llm = ChatOllama(model="llama3")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    print("\n🤖 Ask anything about your codebase. Type 'exit' to quit.\n")
    while True:
        query = input("🧠 You: ")
        if query.lower() in ["exit", "quit"]:
            break
        try:
            answer = qa_chain.invoke({"query": query})
            print(f"🤖 LLM: {answer}\n")
        except Exception as e:
            print(f"⚠️ Error: {e}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local AI Codebase Assistant")
    parser.add_argument("--codebase", type=str, help="Path to local code folder")
    parser.add_argument("--git", type=str, help="Clone Git repo from URL")
    parser.add_argument("--reindex", action="store_true", help="Force reindex of codebase")

    args = parser.parse_args()
    db_path = "./chroma_db"
    temp_dir = None

    if args.git:
        args.reindex = True
        temp_dir = tempfile.mkdtemp(prefix="temp_cloned_")
        code_path = clone_git_repo(args.git, temp_dir)
    elif args.codebase:
        code_path = Path(args.codebase)
        if not code_path.exists():
            print(f"❌ Codebase not found: {code_path}")
            exit(1)
    else:
        print("❗ Please provide either --codebase or --git")
        exit(1)

    if args.reindex or not Path(db_path).exists():
        vectordb = build_vectorstore(code_path, db_path)
    else:
        vectordb = Chroma(
            persist_directory=db_path,
            embedding_function=OllamaEmbeddings(model="nomic-embed-text")
        )

    try:
        start_chat(vectordb)
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
            print("🧹 Cleaned up temporary cloned repo.")
