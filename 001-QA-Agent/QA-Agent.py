import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# ---------------------------
# Step 1: Get OpenAI Key from Environment Variable
# ---------------------------
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

os.environ["OPENAI_API_KEY"] = openai_api_key  # ensures all libraries can access it

# ---------------------------
# Step 2: Load and Split the Document
# ---------------------------
loader = TextLoader("example.txt")  # Replace with your actual file path
documents = loader.load()

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# ---------------------------
# Step 3: Create Embeddings and Vector Store
# ---------------------------
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)

# ---------------------------
# Step 4: Create the Retrieval-Based QA Chain
# ---------------------------
retriever = db.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0),
    retriever=retriever,
    return_source_documents=True
)

# ---------------------------
# Step 5: Ask Questions
# ---------------------------
query = "What is the document about?"
result = qa_chain(query)

print("Answer:", result["result"])
print("\nSources:\n")
for doc in result["source_documents"]:
    print(doc.page_content[:300] + "\n---")  # preview the first 300 characters
