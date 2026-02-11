import pandas as pd
import os

import lotus
from lotus.models import LM
from lotus.types import ReasoningStrategy

# prepare RITS setting
RITS_API_KEY = os.environ.get("RITS_API_KEY")
rits_model_name = "meta-llama/llama-3-3-70b-instruct"
rits_model_url = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-3-70b-instruct/v1"

base_config = {}
base_config['api_base'] = rits_model_url
base_config['extra_headers'] = {"RITS_API_KEY": RITS_API_KEY}

lm = LM(model='openai/' + rits_model_name, **base_config)

lotus.settings.configure(lm=lm)


# Test filter operation on an easy dataframe
data = {
    "Text": [
        "I had two apples, then I gave away one",
        "My friend gave me an apple",
        "I gave away both of my apples",
        "I gave away my apple, then a friend gave me his apple, then I threw my apple away",
    ]
}
df = pd.DataFrame(data)
user_instruction = "{Text} I have at least one apple"
# filtered_df = df.sem_filter(user_instruction, strategy="cot", return_all=True)
filtered_df = df.sem_filter(
    user_instruction, strategy=ReasoningStrategy.ZS_COT, return_all=True, return_explanations=True
)  # uncomment to see reasoning chains

print(filtered_df)
# print(filtered_df)
