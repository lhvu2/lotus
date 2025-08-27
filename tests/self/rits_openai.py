from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import os


class Joke(BaseModel):
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")


llm = ChatOpenAI(
    model="ibm-granite/granite-3.1-8b-instruct",
    temperature=0,
    max_retries=2,
    api_key='/',
    base_url='https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-1-8b-instruct/v1',
    default_headers={'RITS_API_KEY': os.environ["RITS_API_KEY"]},
)

structured_llm = llm.with_structured_output(Joke, method='json_schema')
response = structured_llm.invoke("Tell me a joke about cats")
print(response)

pass