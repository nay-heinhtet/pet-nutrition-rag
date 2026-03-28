from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

loader = DirectoryLoader('docs', glob='*.pdf', loader_cls=PyPDFLoader)
documents = loader.load()
print(f"Loaded {len(documents)} pages")
