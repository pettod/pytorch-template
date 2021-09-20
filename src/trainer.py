from src.utils.base_trainer import Basetrainer
import src.utils.utils as ut


class Trainer(Basetrainer):
    def forwardPropagation(self, batch):
        x, y = ut.toDevice(batch)
        prediction = self.model(x)
        loss = self.loss_function(prediction, y)
        return prediction, y, loss

    def validationIteration(self, batch):
        return self.forwardPropagation(batch)

    def trainIteration(self, batch):
        prediction, y, loss = self.forwardPropagation(batch)
        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()
        return prediction, y, loss
