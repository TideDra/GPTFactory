from typing import Optional
import time
from threading import Lock
class LimitationChecker:
    """A checker to check whether current API usage exceeds the limitation.
    """
    def __init__(self,token_rate_limit:Optional[int] = None, request_rate_limit:Optional[int] = None) -> None:
        """Initialize the checker.

        Args:
            token_rate_limit (int): token number limit per minute.
            request_rate_limit (_type_): request number limit per minute.
        """
        assert token_rate_limit is not None or request_rate_limit is not None, "At least one limitation is needed!"
        self.token_rate_limit = token_rate_limit
        self.request_rate_limit = request_rate_limit
        self.delay = 60 / request_rate_limit
        self.request_history = []
        self.token_history = []
        self.lock = Lock()
    
    def record_token(self,request_time:int,used_token_num:int = None):
        """Record the usage.

        Args:
            used_token_num (int): token number used in the request.
        """
        with self.lock:
            self.__clean_old_history()
            if self.token_rate_limit is None:
                assert used_token_num is None, "You're recording used_token_num, but token_rate_limit is not set!"
            self.token_history.append((request_time,used_token_num))

    def record_request(self,request_time:int):
        """Record the usage.

        Args:
            used_token_num (int): token number used in the request.
        """
        with self.lock:
            self.__clean_old_history()
            self.request_history.append(request_time)
    
    def __clean_old_history(self):
        """Clean the history that is older than 60 seconds.
        """
        now = time.time()
        self.token_history = [item for item in self.token_history if now - item[0] < 60]
        self.request_history = [item for item in self.request_history if now - item < 60]

    def __check_if_reach_limitation(self) -> bool:
        """Check whether the usage exceeds the limitation.

        Returns:
            bool: True if the usage exceeds the limitation.
        """
        self.__clean_old_history()
        if self.token_rate_limit is not None:
            token_num = sum([item[1] for item in self.token_history if item[1] is not None])
            if token_num >= self.token_rate_limit:
                #logger.info(f"Token number {token_num} exceeds the limitation {self.token_rate_limit}!")
                return True
        if self.request_rate_limit is not None:
            if len(self.request_history) == 0:
                return False
            last_request_time = self.request_history[-1]
            now = time.time()
            if now - last_request_time < self.delay:
            #logger.info(f"Request number {request_num} exceeds the limitation {self.request_rate_limit}!")
                return True
        return False

    def wait(self):
        """Wait until the usage is below the limitation.
        """
        with self.lock:
            if self.__check_if_reach_limitation():
                last_request_time = self.request_history[-1]
                now = time.time()
                sleep_time = max(self.delay - (now - last_request_time),self.delay)
                #logger.info(f"Sleep {sleep_time} seconds to wait for the limitation.")
                time.sleep(sleep_time)
