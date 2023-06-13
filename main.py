from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from transcriber import audio_to_text



file_path = "cancel_2.wav"
audio_transcript = audio_to_text(file_path)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splitted_text = text_splitter.split_text(audio_transcript)
embeddings = OpenAIEmbeddings()
audio_search = Chroma.from_texts(splitted_text, embeddings).as_retriever()


query = "What is name and DOB of customer?"
docs = audio_search.get_relevant_documents(query)

chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
output = chain.run(input_documents=docs, question=query)
print(output)