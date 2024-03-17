import os
import sys
sys.path.extend([os.getcwd()]) #TODO: take out before sending

GOOGLE_API_KEY = "INSERT_API_KEY"
SEI = "INSERT_SEI_KEY"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["GOOGLE_CSE_ID"] = SEI

os.environ["OAI_CONFIG_LIST"] = """[
    {
        "model": "gpt-4-0125-preview",
        "api_key": "INSERT OPENAI API KEY"
    }
]"""

OPENAI_KEY_ME = "INSERT OPENAI API KEY"
os.environ['OPENAI_API_KEY'] = OPENAI_KEY_ME

import os
import agents
import autogen.runtime_logging
import random
import string
import tiktoken
from datetime import datetime

LOGS_PATH = "logs/"

def simulate(round_number, total_days):
    current_date = datetime.now().strftime("%m%d%y")
    random_sequence = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
    db_filename = os.path.join(LOGS_PATH, f"{round_number}_{random_sequence}_{current_date}_logs.db")
    chat_filename = os.path.join(LOGS_PATH, f"{round_number}_chat_output_{random_sequence}_{current_date}.txt")
    groupchat, manager, user_proxy = agents.initialize()
    message = f"Simulate round number {round_number} for the planet comprising {total_days} days."
    autogen.runtime_logging.start(logger_type="sqlite", config={"dbname": db_filename})
    user_proxy.initiate_chat(manager, message=message)
    autogen.runtime_logging.stop()
    agents.save_chat(groupchat, chat_filename)
    return True

def count_tokens(st, model="openai"):
    if model == "openai":
        encoding = tiktoken.encoding_for_model("gpt-4-0125-preview")
        num_tokens = len(encoding.encode(st))
        return num_tokens
    else:
        return "Not implemented"
