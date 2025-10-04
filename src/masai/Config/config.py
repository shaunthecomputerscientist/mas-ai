


class Config:
    def __init__(self):
        self.MAX_RECURSION_LIMIT: int = 100
        self.stream_mode: str = "updates"
        self.max_tool_loops: int = 3
        self.truncated_response_length: int = 500
        self.MAX_REFLECTION_COUNT: int = 3
config = Config()