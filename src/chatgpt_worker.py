import re
import time
import httpx
from typing import List


from openai import OpenAI
from openai.types.chat import ChatCompletion

PROXY = 'socks5://49.13.59.71:8443'
AI_REQUEST_RETRY = 3


class GPTRequest:
    def __init__(self,
                 model='gpt-4-turbo',
                 temperature=1.0,
                 max_tokens=900,
                 top_p=1,
                 frequency_penalty=0,
                 presence_penalty=0,
                 stop='<CHATGPT_STOP_GEN>',
                 timeout=120,
                 response_format=None):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop
        self.timeout = timeout
        self.response_format = response_format if response_format is not None else {"type": "text"}

        self._role_system = dict()
        self._role_assistant = dict()
        self._role_user: List[dict] = list()

    def add_role_system(self, content: str) -> None:
        if len(self._role_system) != 0:
            raise Exception('System role already exists')
        self._role_system = {
            "role": "system",
            "content": "{}".format(content)
        }

    def add_role_assistant(self, content: str) -> None:
        if len(self._role_assistant) != 0:
            raise Exception('Assistant role already exists')
        self._role_assistant = {
            "role": "assistant",
            "content": "{}".format(content)
        }

    def add_role_user(self, content: str) -> None:
        self._role_user.append({
            "role": "user",
            "content": "{}".format(content + '\n' + self.stop)
        })

    def get_messages(self) -> List[dict]:
        messages = list()
        if len(self._role_system) != 0:
            messages.append(self._role_system)
        if len(self._role_assistant) != 0:
            messages.append(self._role_assistant)
        if len(self._role_user) == 0:
            raise Exception('User role not set')
        messages += self._role_user
        return messages

    def get_params(self) -> dict:
        return {
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'top_p': self.top_p,
            'frequency_penalty': self.frequency_penalty,
            'presence_penalty': self.presence_penalty,
            'stop': self.stop,
            'timeout': self.timeout,
            'response_format': self.response_format
        }

    def clear(self):
        self._role_system.clear()
        self._role_assistant.clear()
        self._role_user.clear()


def ask_chatgpt(chatgpt_request: GPTRequest) -> str:
    http_client = httpx.Client(proxy=PROXY)
    openai_client = OpenAI(
        api_key='sk-szBlb9gExPwLCiHBovKNy4s0PJ7CUXTwk-OBjZimmFT3BlbkFJdArRsxVRYnvCKAyGG7-kXOXkNxylJEZ164qg9lJqkA',
        http_client=http_client
    )
    response = None
    for i in range(AI_REQUEST_RETRY):
        try:
            response = openai_client.chat.completions.create(
                messages=chatgpt_request.get_messages(),
                **chatgpt_request.get_params()
            )
        except Exception as ex:
            print('Try {}. Assistant request error. Sleep for 5 sec. err="{}"'.format(i, ex))
            time.sleep(5)
            continue
        break
    if response is None:
        raise Exception("Assistant no answer or an empty response.")

    if (
            isinstance(response, list) and
            len(response) == 2 and
            isinstance(response[1], ChatCompletion) and
            'reduce the length of the messages' in response[1].body
    ):
        raise Exception(f'Model get too long text. Model response: "{response[1].body}"')

    if (
            response.choices is None
            or len(response.choices) == 0
            or response.choices[0].message is None
            or response.choices[0].message.content is None
    ):
        raise Exception("Assistant no answer")

    if "error" in response:
        raise Exception('Assistant return error: "{}"'.format(response))

    return response.choices[0].message.content


def create_rephrase(text: str, num_variants: int=5) -> List[str]:
    request = GPTRequest()
    request.add_role_system(
        "Your are cybersecurity assistant. You know all about MITRE ATTCK techniques and tactics."
    )
    request.add_role_user(
        'Your task is to generates multiple rephrased alternative variants based on a '
        f'single input text. Generate {num_variants} variant of text, one on each line, '
        'related to the input text below. Include in your answer only variants of text:\n'
        f'{text}\n'
    )

    response = ask_chatgpt(request)

    return re.sub('\n+','\n', response).strip(' \r\t\n').split('\n')