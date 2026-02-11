import pandas as pd
import os

import lotus
from lotus.models import LM
from lotus.types import ReasoningStrategy

# IBM Litellm Proxy
model_name = "aws/claude-haiku-4-5"

base_config = {}
base_config['api_base'] = "https://ete-litellm.ai-models.vpc-int.res.ibm.com"
base_config['api_key'] = os.environ.get("IBM_LITELLM_API_KEY")

lm = LM(model='openai/' + model_name, **base_config)

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