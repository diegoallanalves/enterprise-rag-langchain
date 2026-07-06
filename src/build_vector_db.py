import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from config import ORDERS_FILE, INVENTORY_FILE, POLICY_FILE, CHROMA_DIR, OPENAI_EMBEDDING_MODEL

def create_documents():
    orders = pd.read_csv(ORDERS_FILE)
    inventory = pd.read_csv(INVENTORY_FILE)
    policy = POLICY_FILE.read_text(encoding="utf-8")
    documents = [Document(page_content=policy, metadata={"source": "business_policy"})]

    merged = orders.merge(inventory, on=["Product_ID", "Product_Name"], how="left")
    for _, row in merged.iterrows():
        text = (
            f"PO {row['PO_Number']} customer {row['Customer_Name']} ordered "
            f"{row['Quantity_Ordered']} units of {row['Product_Name']}. "
            f"Current stock is {row['Current_Stock']}. "
            f"Order status is {row['Order_Status']}. Region is {row['Region']}."
        )
        documents.append(Document(page_content=text, metadata={"source": "sales_record", "po_number": row["PO_Number"]}))
    return documents

def build_vector_db():
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
    db = Chroma.from_documents(create_documents(), embedding=embeddings, persist_directory=str(CHROMA_DIR))
    print(f"Vector database created at {CHROMA_DIR}")
    return db

if __name__ == "__main__":
    build_vector_db()
