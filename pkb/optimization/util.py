import os


class Logger:
    def __init__(self, filename="logs/optimization_log.txt"):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.filename = filename

        with open(self.filename, 'w') as f:
            f.write("Optimization Log\n")

    def log(self, message):
        with open(self.filename, 'a') as f:
            f.write(message + "\n")