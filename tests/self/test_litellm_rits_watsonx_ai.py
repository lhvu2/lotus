# source: https://docs.litellm.ai/docs/providers/watsonx

import os
from litellm import completion

url = os.environ["WATSONX_URL"]
key = os.environ["WATSONX_APIKEY"]
proj = os.environ["WATSONX_PROJECT_ID"]

## Call WATSONX `/text/chat` endpoint - supports function calling
response = completion(
  model="watsonx/meta-llama/llama-3-2-3b-instruct",
  messages=[{ "content": "what is your favorite colour?","role": "user"}],
  project_id=proj # or pass with os.environ["WATSONX_PROJECT_ID"]
)

## Call WATSONX `/text/generation` endpoint - not all models support /chat route. 
response = completion(
  model="watsonx/ibm/granite-3-3-8b-instruct",
  messages=[{ "content": "what is your favorite colour?","role": "user"}],
  project_id=proj
)

print("WXAI .............................")
print(response)

# source: https://ibm-research.slack.com/archives/C07PNSBQMQW/p1748955760919669?thread_ts=1748887035.578709&cid=C07PNSBQMQW
from litellm import completion
import os

## set ENV variables
os.environ["RITS_API_KEY"] = os.environ["RITS_API_KEY"]
os.environ["OPENAI_API_KEY"] = "xxxxxx"

# openai must prefix the model name to let litellm know that it's openai provider and "chat/completions" is desired
#
# leave off "chat/completions" as this is assumed by 'openai'
#
# for 'completions' use text-completion-openai/ in front of model
#

response = completion(
      model="openai/ibm/slate-125m-english-rtrvr-v2",
      messages=[{"role": "system","content": "You are Granite Chat, ....."},
           {"role": "user","content": "Write a haiku about generative design"}],
      api_base="https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/slate-125m-english-rtrvr-v2/v1/",
      extra_headers={'RITS_API_KEY': os.environ["RITS_API_KEY"]}

)

print("RITS .............................")
print(response)

pass
