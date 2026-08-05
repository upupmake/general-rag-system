class ResponseWrapper:
    def __init__(self, content: str | list, response_metadata: dict = None):
        self.content = content
        self.response_metadata = response_metadata

    def __repr__(self):
        return f"ResponseWrapper(content='{self.content}')"
