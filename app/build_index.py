from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import glob
from yandex_chain import YandexEmbeddings

def file_to_chunk(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

        # Разделение текста на чанки
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )

        chunks = text_splitter.split_text(text)
        print(f"Создано {len(chunks)} чанков")

        return chunks

def create_vectorstore(chunks):
    print("Создание векторного хранилища...")
    embeddings = YandexEmbeddings(config="config.json")
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    vectorstore.save_local("faiss.index")

print("Загрузка документов...")
chunks = []
for file in glob.glob("./knowledge_base/*.txt"):
    chunks.extend(file_to_chunk(file))
print(len(chunks))
create_vectorstore(chunks)

