class Error(Exception):
    def __init__(self, expr = None, msg = None):
        self.expr = expr
        self.msg = msg
class inputSmallerThanKernel(Error):
    def __init__(self):
        super(inputSmallerThanKernel, self).__init__()
class nodeDoesNotExist(Error):
    def __init__(self):
        super(nodeDoesNotExist, self).__init__()