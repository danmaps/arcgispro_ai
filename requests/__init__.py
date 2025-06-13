class RequestException(Exception):
    pass

class exceptions:
    RequestException = RequestException

def post(*args, **kwargs):
    raise RequestException("requests module not available")

def get(*args, **kwargs):
    raise RequestException("requests module not available")
