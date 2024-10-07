class Error_handler:
    def __init__(self, verbose=1):
        self.verbose = verbose

    def __call__(self, code, message):
        if self.verbose:
            print(f'*** ERROR {code}: {message}')

# Define specific error codes as constants
class ErrorCode:
    DOMAIN_ERROR = 1
    COMMUNICATION_ERROR = 2

class Log:
    def __init__(self, verbose):
        self.verbose_level = verbose

    def __call__(self, message, verbose):
        if verbose:
            print(message)