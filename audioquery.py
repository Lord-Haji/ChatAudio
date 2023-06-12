import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFium2Loader
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from transcriber import audio_to_text


class AudioQuery:
    def __init__(self, openai_api_key = None) -> None:
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
        self.chain = None
        self.db = None

    def ask(self, question: str) -> str:
        if self.chain is None:
            response = "Please, add a file."
        else:
            docs = self.db.get_relevant_documents(question)
            response = self.chain.run(input_documents=docs, question=question)
        return response

    def ingest(self, file_path: os.PathLike) -> None:

        audio_transcript = audio_to_text(file_path)

        splitted_text = self.text_splitter.split_text(audio_transcript)

        self.db = Chroma.from_texts(splitted_text, self.embeddings).as_retriever()
        self.chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")

    def forget(self) -> None:
        self.db = None
        self.chain = None