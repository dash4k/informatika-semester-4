import sys

class Spinner:
    def __init__(self, message="Processing"):
        self.spinner = ['|', '/', '-', '\\']
        self.idx = 0
        self.message = message

    def spin(self):
        sys.stdout.write(f"\r{self.message} {self.spinner[self.idx % len(self.spinner)]}")
        sys.stdout.flush()
        self.idx += 1