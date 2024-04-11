from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os
import pandas as pd

df = pd.read_csv(r"C:\Users\jyjun\git\Langchain_ChatModel_Test\대구광역시_관광지_160개.csv", encoding="cp949")

# OpenAI API 키 설정
os.environ['openai_api_key'] =''

llm = ChatOpenAI()

prompt = ChatPromptTemplate.from_messages([("system" ,'''Your name is “가볼까?” He is a Daegu travel recommendation expert 
                                            who recommends Daegu travel destinations. You must always answer in Korean.
                                            You must speak kindly.
                                            1. Destination
                                            2. Destination
                                            All you have to do is provide 5 destinations in this format.'''),
                                           ("user", "{message}")])

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

# 사용자 입력과 채팅 기록을 관리하는 함수
def response(message, history):
    history_langchain_format = []
    for msg in history:
        if isinstance(msg, HumanMessage):
            history_langchain_format.append(msg)
        elif isinstance(msg, AIMessage):
            history_langchain_format.append(msg)

    # 새로운 사용자 메시지 추가
    history_langchain_format.append(HumanMessage(content=message))

    # LangChain ChatOpenAI 모델을 사용하여 응답 생성
    gpt_response = chain.invoke({"message" : message})

    # 생성된 AI 메시지를 대화 이력에 추가
    history_langchain_format.append(AIMessage(content=gpt_response))

    return gpt_response, history_langchain_format

# 대화 이력 초기화
chat_history = []

#추천 모델

#불용어 처리
korean_stop_words = [
    "이", "그", "저", "에", "가", "을", "를", "의", "은", "는", "들", "를", "과", "와", "에게", "게",
    "합니다", "하는", "있습니다", "합니다", "많은", "많이", "많은", "많이", "모든", "모두", "한", "그리고", "그런데",
    "나", "너", "우리", "저희", "이런", "그런", "저런", "어떤", "어느", "그럴", "것", "그것", "이것", "저것", 
    "그러나", "그리하여", "그러므로", "그래서", "하지만", "그럼에도", "이에", "때문에", "그래서", "그러니까", 
    "이렇게", "그렇게", "저렇게", "어떻게", "왜", "무엇", "어디", "언제", "어떻게", "어느", "모두", "모든", 
    "그래도", "하지만", "그러면", "그런데", "하지만", "이러한", "그러한", "저러한", "이러한", "이렇게", "그렇게",
    "저렇게", "어떻게", "왜", "어디", "언제", "어떻게", "모두", "모든", "몇", "누구", "무슨", "어느", "얼마나",
    "무엇", "무슨", "아무", "여기", "저기", "거기", "그곳", "이곳", "저곳", "무엇", "아무", "모두", "마치",
    "보다", "보이다", "등", "등등", "등등등"
    ]

# def recommend(df, user_input, korean_stop_words):
#     all_about_data = df['all about'].tolist()

#     tfidf = TfidfVectorizer(stop_words=korean_stop_words)
#     tfidf_matrix_input = tfidf.fit_transform(user_input)
#     tfidf_matrix_all_about = tfidf.transform(all_about_data)

#     # 코사인 유사도 계산
#     cosine_sim = linear_kernel(tfidf_matrix_input, tfidf_matrix_all_about)

#     # 상위 5개의 유사한 관광지 이름 출력
#     top_place = cosine_sim.argsort()[0][-2:][::-1]

#     for i, idx in enumerate(top_place, 5):
#         print(f"{i}. {df['name'][idx]}")

def recommend(df, user_input, korean_stop_words):
    user_input_list = [user_input]
    
    all_about_data = df['all about'].tolist()

    tfidf = TfidfVectorizer(stop_words=korean_stop_words)
    tfidf_matrix_all_about = tfidf.fit_transform(all_about_data)
    tfidf_matrix_input = tfidf.transform(user_input_list)

    # 코사인 유사도 검사
    cosine_sim = linear_kernel(tfidf_matrix_input, tfidf_matrix_all_about)

    top_place = cosine_sim.argsort()[0][-5:][::-1]

    for rank, idx in enumerate(top_place, start=1):
        print(f"{rank}. {df.iloc[idx]['name']}")



# 대화 시작
print("대화를 시작합니다. 대화를 종료하려면 '고마워'라고 입력하세요.")

while True:
    user_input = input("당신: ")
    if user_input.lower() == '고마워':
        print("대화를 종료합니다. 좋은 하루 되세요 😊")
        break

    # 응답 생성 및 출력
    ai_response, chat_history = response(user_input, chat_history)
    print("가볼까:", ai_response)
    recommend(df, user_input, korean_stop_words)    
