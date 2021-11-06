from PIL import Image

from config import CONFIG
from src.utils.base_trainer import Basetrainer
import src.utils.utils as ut


class Trainer(Basetrainer):
    def forwardPropagation(self, batch):
        x, y = ut.toDevice(batch)
        prediction = self.models[0](x)
        loss = self.loss_function(prediction, y)
        return prediction, y, loss

    def validationIteration(self, batch):
        return self.forwardPropagation(batch)

    def trainIteration(self, batch):
        prediction, y, loss = self.forwardPropagation(batch)
        self.models[0].zero_grad()
        loss.backward()
        self.optimizers[0].step()
        return prediction, y, loss

    def testAfterEpoch(self):
        x = Image.open(CONFIG.TEST_IMAGE_PATH).convert("RGB")
        x = CONFIG.INPUT_NORMALIZE(CONFIG.TEST_TRANSFORM(x))
        return self.models[0](x.unsqueeze(0)).squeeze(0)
