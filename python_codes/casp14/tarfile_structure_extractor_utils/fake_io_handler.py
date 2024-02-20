class PDBIOFakeFileHandler:

    def __init__(self):
        self.content = ""

    def write(self, other: str):
        self.content += other

    def close(self):
        pass
