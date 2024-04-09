from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# OpenAI API 키 설정
# llm = ChatOpenAI(openai_api_key='')

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
