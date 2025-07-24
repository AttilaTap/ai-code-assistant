import argparse
import shutil
import tempfile
from pathlib import Path
import subprocess
import hashlib

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document



def clone_git_repo(git_url: str, clone_dir: str) -> Path:
    print(f"📥 Cloning repo from: {git_url}")
    repo_path = Path(clone_dir) / Path(git_url).stem
    subprocess.run(["git", "clone", git_url, str(repo_path)], check=True)
    return repo_path

def safe_load_documents(code_path: Path):
    all_docs = []
    skip_dirs = {"node_modules", ".git", ".venv", "venv", "__pycache__", "dist", "build"}
    excluded_extensions = {
        ".md", ".txt", ".lock", ".config", ".yml", ".yaml", ".json",
        ".csv", ".tsv", ".jpg", ".jpeg", ".png", ".ico", ".gif", ".svg",
        ".env", ".log", ".toml", ".ini"
    }

    for file_path in code_path.rglob("*"):
        if file_path.is_file():
            # Skip unwanted directories
            if any(part in skip_dirs for part in file_path.parts):
                continue

            # Skip excluded extensions
            if file_path.suffix.lower() in excluded_extensions:
                continue

            try:
                loader = TextLoader(str(file_path), encoding="utf-8")
                all_docs.extend(loader.load())
            except UnicodeDecodeError as e:
                print(f"⚠️ Skipping binary or unreadable file: {file_path} ({e})")
            except Exception as e:
                print(f"⚠️ Skipping file due to unexpected error: {file_path} ({e})")

    return all_docs

def get_available_ollama_models():
    try:
        output = subprocess.check_output(["ollama", "list"], encoding="utf-8")
        models = []
        for line in output.splitlines()[1:]:  # Skip header
            parts = line.split()
            if parts:
                models.append(parts[0])
        return models
    except Exception as e:
        print(f"⚠️ Failed to list Ollama models: {e}")
        return []


def build_vectorstore(code_path: Path, persist_dir: str = "./chroma_db"):
    from langchain_community.vectorstores import Chroma
    import os

    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print(f"📦 Loading existing vectorstore from: {persist_dir}")
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        return vectordb

    print(f"\n📂 Loading and indexing files from: {code_path}")
    docs = safe_load_documents(code_path)
    print(f"📁 Total files loaded: {len(docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
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
    retriever = vectordb.as_retriever(search_kwargs={"k": 8})
    llm = ChatOllama(model="codellama:34b-instruct")
    system_prompt = load_prompt_template()

    if system_prompt:
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=system_prompt
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={
                "prompt": prompt,
                "document_variable_name": "context"
            }
        )
    else:
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    print("\n🤖 Ask anything about your codebase. Type 'exit' to quit.\n")
    while True:
        query = input("🧠 You: ")
        if query.lower() in ["exit", "quit"]:
            break
        try:
            answer = qa_chain.invoke({"query": query})
            print(f"🤖 LLM: {answer['result']}\n")
        except Exception as e:
            print(f"⚠️ Error: {e}\n")
            
def load_prompt_template(file_path="prompt.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print("⚠️ prompt.txt not found. Using default behavior.")
        return None

def create_snippet_chain(snippet: str, model_name: str = "codellama:34b-instruct"):
    doc = Document(page_content=snippet)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectordb = Chroma.from_documents([doc], embedding=embeddings)
    
    retriever = vectordb.as_retriever(search_kwargs={"k": 1})
    llm = ChatOllama(model=model_name)
    system_prompt = load_prompt_template()
    print("System prompt loaded:\n", system_prompt)

    if system_prompt:
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=system_prompt
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={
                "prompt": prompt,
                "document_variable_name": "context"
            }
        )
    else:
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    return qa_chain
    
import hashlib
from pathlib import Path
import tempfile
import shutil

def load_codebase_and_start_chat(git_url=None, code_path=None, model_name="codellama:13b-instruct"):
    print(f"⚙️ Starting to load codebase. git_url={git_url}, code_path={code_path}, model={model_name}")
    llm = ChatOllama(model=model_name)
    temp_dir = None

    try:
        # Determine path and generate a unique persist_name
        if git_url:
            temp_dir = tempfile.mkdtemp(prefix="temp_cloned_")
            code_path = clone_git_repo(git_url, temp_dir)
            persist_name = hashlib.md5(git_url.encode()).hexdigest()
        elif code_path:
            code_path = Path(code_path)
            if not code_path.exists():
                raise FileNotFoundError(f"❌ Local path not found: {code_path}")
            persist_name = hashlib.md5(str(code_path.resolve()).encode()).hexdigest()
        else:
            raise ValueError("❌ You must provide either a git_url or a code_path.")

        # Use unique persistence path
        persist_path = f"./chroma_db/{persist_name}"
        vectordb = build_vectorstore(code_path, persist_dir=persist_path)

        retriever = vectordb.as_retriever(search_kwargs={"k": 8})
        system_prompt = load_prompt_template()

        if system_prompt:
            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=system_prompt
            )
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                return_source_documents=False,
                chain_type_kwargs={
                    "prompt": prompt,
                    "document_variable_name": "context"
                }
            )
        else:
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        print("✅ QA chain successfully created.")
        return qa_chain

    except Exception as e:
        print(f"❌ Error in load_codebase_and_start_chat: {e}")
        return None

    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
            print("🧹 Cleaned up temporary cloned repo.")


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
