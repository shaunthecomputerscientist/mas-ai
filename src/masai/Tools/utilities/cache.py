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

# Create an instance of RedisCache with your credentials.
# redis_cache = MASAIToolCache(host='localhost', port=6379, db=0)
# Decorate the function with both the caching and the @tool decorators.
# @redis_cache.func  # Outer decorator: handles caching.
# @tool            # Inner decorator: registers the function as a LangChain tool.
# def my_function(x, y):
#     return x + y

# # Example usage:
# print(my_function(2, 3))  # Will compute, cache, and print the result (5).

