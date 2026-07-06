from rag_chain import run_rag_query

question = input("Ask a business question: ")
answer, docs, _ = run_rag_query(question)

print("\nAI Answer")
print("-" * 60)
print(answer)

print("\nRetrieved Sources")
print("-" * 60)
for doc in docs:
    print(doc.metadata)
