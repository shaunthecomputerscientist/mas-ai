import redis
import functools
import pickle
from langchain.tools import tool  # Importing the @tool decorator

class ToolCache:
    def __init__(self, host='localhost', port=6379, db=0, password=None):
        # Establish a connection to the Redis server with the given credentials.
        """
            Creates tool cache for MAS Ai tools.
        Args:
            host (str, optional): Host of redis service. Defaults to 'localhost'.
            port (int, optional): Port of redis service. Defaults to 6379.
            db (int, optional): database name. Defaults to 0.
            password (_type_, optional): password of redis service. Defaults to None.
        """
        self.redis = redis.Redis(host=host, port=port, db=db, password=password)

    def masai_cache(self, func):
        """
        This method returns a decorator that caches the result of the function using Redis.
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate a unique key based on the function name and its arguments.
            key = f"{func.__name__}:{args}:{kwargs}"
            cached_result = self.redis.get(key)
            if cached_result is not None:
                # Return the cached result if available.
                return pickle.loads(cached_result)
            # Otherwise, call the original function and cache its result.
            result = func(*args, **kwargs)
            self.redis.set(key, pickle.dumps(result))
            return result
        return wrapper