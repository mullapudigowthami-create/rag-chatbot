import os
import glob
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ── Constants ──────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL        = "llama-3.3-70b-versatile"
CHUNK_SIZE   = 500
CHUNK_OVERLAP = 50

# ── Helpers ────────────────────────────────────────────────────────────────────

def list_pdfs() -> list[str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return glob.glob(os.path.join(script_dir, "*.pdf"))


def resolve_pdf(user_input: str) -> str | None:
    user_input = user_input.strip().strip('"').strip("'")
    if os.path.isfile(user_input):
        return os.path.abspath(user_input)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(script_dir, user_input)
    if os.path.isfile(candidate):
        return candidate
    for p in list_pdfs():
        if user_input.lower() in os.path.basename(p).lower():
            return p
    return None


def build_rag_chain(pdf_path: str):
    print("\n⏳ Loading PDF …")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    if not documents:
        raise ValueError("No content found in PDF.")
    print(f"✅ Loaded {len(documents)} page(s).")

    print("⏳ Splitting into chunks …")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks.")

    print("⏳ Building vector store (first run may take a moment) …")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    print("✅ Vector store ready.")

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=MODEL,
    )

    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question based ONLY on the context below.
If the answer is not in the context, say "I couldn't find that in the document."

Context:
{context}

Question: {question}
""")

    rag_chain = (
        {"context": retriever, "question": lambda x: x}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("       🤖  RAG Chatbot  (Groq + FAISS)")
    print("=" * 55)

    # Show available PDFs
    available = list_pdfs()
    if available:
        print("\n📂 PDFs found in folder:")
        for p in available:
            print(f"   • {os.path.basename(p)}")
        print()

    # Get PDF path
    while True:
        raw = input("Enter PDF filename (or full path): ").strip()
        if not raw:
            continue
        pdf_path = resolve_pdf(raw)
        if pdf_path:
            print(f"✅ Using: {pdf_path}")
            break
        else:
            print(
                f"❌ Could not find '{raw}'.\n"
                "   • Make sure the PDF is in the same folder as this script.\n"
            )

    # Build RAG chain
    try:
        rag_chain = build_rag_chain(pdf_path)
    except Exception as e:
        print(f"\n❌ Failed to build RAG chain: {e}")
        return

    # Chat loop
    print("\n" + "─" * 55)
    print("💬 Ask questions about the document. Type 'exit' to quit.")
    print("─" * 55 + "\n")

    while True:
        try:
            question = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 👋")
            break

        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye! 👋")
            break
        if not question:
            continue

        try:
            answer = rag_chain.invoke(question)
            print(f"\n🤖 Answer: {answer}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    main()