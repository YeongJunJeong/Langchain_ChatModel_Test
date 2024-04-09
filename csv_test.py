from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
import os

df = pd.read_csv(r"C:\Users\jyjun\git\Langchain_ChatModel_Test\대구광역시_관광지_160개.csv", encoding = "cp949")
selected_columns = ['name','classification', 'street_name_adress', 'tag', 'time']
df_selected = df[selected_columns]

os.environ['openai_api_key'] =''

agent = create_pandas_dataframe_agent(ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
                         df_selected,
                         verbose = True,
                         agent_type = AgentType.OPENAI_FUNCTIONS,
)

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
    gpt_response = agent.run(message)
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


