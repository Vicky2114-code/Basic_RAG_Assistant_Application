import time
from langchain.callbacks.base import BaseCallbackHandler
from constant import TIMEOUT_SECONDS, TEMPLATES

class SmartStreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
        self.start_time = time.time()

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text + "▌")

        if time.time() - self.start_time > TIMEOUT_SECONDS:
            raise TimeoutError(TEMPLATES["timeout"])
