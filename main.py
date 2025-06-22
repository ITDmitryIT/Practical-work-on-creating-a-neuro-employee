import os
from getpass import getpass
from langchain import OpenAI, VectorDBQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts import PromptTemplate

# Запрос ручного ввода API ключа
api_key = getpass("Введите ваш API ключ OpenAI: ")
os.environ["OPENAI_API_KEY"] = api_key

# Инициализация модели и эмбеддингов
llm = OpenAI(model_name="gpt-3.5-turbo-instruct")
embeddings = OpenAIEmbeddings()

# Загрузка базы знаний
loader = TextLoader('programming_knowledge.txt')  # Предположим, что у нас есть файл с текстовой информацией
documents = loader.load()
db = FAISS.from_documents(documents, embeddings)

# Определение шаблона запроса
prompt_template = """Вы являетесь опытным программистом-разработчиком.
Ваша задача - ответить на вопрос пользователя, основываясь только на предоставленной информации. Не добавляйте информацию, которой нет в тексте.
Предоставленная информация: {summaries}
Вопрос: {question}
Ответ:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["summaries", "question"])

# Создание цепочки (chain)
chain = load_qa_with_sources_chain(llm, chain_type="stuff", prompt=PROMPT)

# Трассировка работы нейро-сотрудника
query = "Как создать REST API на Python?"
result = chain({"input_documents": documents, "question": query}, return_only_outputs=True)
print(result['output_text'])

# Функция фильтрации запросов
def filter_query(query):
    forbidden_words = ["плохо", "не верно", "ошибка"]
    for word in forbidden_words:
        if word in query.lower():
            return False
    return True

# Пример использования фильтрации запросов
query = "Как создать REST API на Python?"
if filter_query(query):
    result = chain({"input_documents": documents, "question": query}, return_only_outputs=True)
    print(result['output_text'])
else:
    print("Запрос содержит запрещенные слова.")
