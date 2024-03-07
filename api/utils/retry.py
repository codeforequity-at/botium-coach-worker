import openai
import random
import time
from .log import getLogger
logger = getLogger('openai_retry')

"""Retry a function with exponential backoff."""
def openai_retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 1,
    jitter: bool = True,
    max_retries: int = 3,
    errors: tuple = (openai.OpenAIError,),
):
    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay

        while True:
            try:
                return func(*args, **kwargs)
            except errors as e:
                num_retries += 1
                if num_retries > max_retries:
                    raise Exception(f"Maximum number of OpenAI reqtries ({max_retries}) exceeded ({str(e)})")
                delay *= exponential_base * (1 + jitter * random.random())
                logger.warn(f"OpenAI request error #{num_retries}/#{max_retries}: {str(e)}, retry after delay: {delay}")

                time.sleep(delay)
            except Exception as e:
                raise e
    return wrapper