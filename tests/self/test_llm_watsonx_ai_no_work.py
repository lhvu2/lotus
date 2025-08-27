import pandas as pd

import litellm

import lotus
from lotus.models import SentenceTransformersRM, LM
from lotus.vector_store import FaissVS

litellm._turn_on_debug()

from langchain_ibm import WatsonxLLM
import os

# Set up your credentials and project details
# It's recommended to set these as environment variables

WATSONX_URL = os.environ["WATSONX_URL"]
WATSONX_APIKEY = os.environ["WATSONX_APIKEY"]
WATSONX_PROJECT_ID = os.environ["WATSONX_PROJECT_ID"]

# Specify the LLM model you want to use
# For example, the latest IBM Granite model
#MODEL_ID = "meta-llama/llama-3-2-3b-instruct"
MODEL_ID = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"

# Set up the LLM parameters
params = {
    "decoding_method": "greedy",
    "min_new_tokens": 10,
    "max_new_tokens": 50
}

# Initialize the watsonx LLM instance
lm = WatsonxLLM(
    model_id=MODEL_ID,
    url=WATSONX_URL,
    project_id=WATSONX_PROJECT_ID,
    apikey=WATSONX_APIKEY,
    params=params
)


# Configure models for LOTUS
#lm = LM(model="gpt-4o-mini")
rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
vs = FaissVS()

lotus.settings.configure(lm=lm, rm=rm, vs=vs)



# Dataset containing courses and their descriptions/workloads
data = [
    (
        "Probability and Random Processes",
        "Focuses on markov chains and convergence of random processes. The workload is pretty high.",
    ),
    (
        "Deep Learning",
        "Fouces on theory and implementation of neural networks. Workload varies by professor but typically isn't terrible.",
    ),
    (
        "Digital Design and Integrated Circuits",
        "Focuses on building RISC-V CPUs in Verilog. Students have said that the workload is VERY high.",
    ),
    (
        "Databases",
        "Focuses on implementation of a RDBMS with NoSQL topics at the end. Most students say the workload is not too high.",
    ),
]
df = pd.DataFrame(data, columns=["Course Name", "Description"])

# Applies semantic filter followed by semantic aggregation
ml_df = df.sem_filter("{Description} indicates that the class is relevant for machine learning.")
tips = ml_df.sem_agg(
    "Given each {Course Name} and its {Description}, give me a study plan to succeed in my classes."
)._output[0]
pass

