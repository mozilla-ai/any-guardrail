from abc import ABC, abstractmethod

class Guardrail(ABC):
    def __init__(self, modelpath):
        self.modelpath=modelpath
        self.model=self.model_instantiation()

    @abstractmethod
    def classify(self):
        raise NotImplementedError("Each subclass will creat their own method.")
    
    @abstractmethod
    def model_instantiation(self):
        raise NotImplementedError("Each subclass will creat their own method.")