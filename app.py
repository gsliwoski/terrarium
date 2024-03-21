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

def simulate(total_days):
    current_date = datetime.now().strftime("%m%d%y")
    history, round_number = agents.load_history()
    round_number += 1
    random_sequence = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
    db_filename = os.path.join(LOGS_PATH, f"{round_number}_{random_sequence}_{current_date}_logs.db")
    chat_filename = os.path.join(LOGS_PATH, f"{round_number}_chat_output_{random_sequence}_{current_date}.txt")
    groupchat, manager, user_proxy = agents.initialize()
    planet_information = agents.get_planet_info()
    message = f"Simulate round number {round_number} for the planet comprising {total_days} days. "
    message += f"The basic planetary information is {planet_information} and the history leading up to the current round of simulation is {history}."
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
