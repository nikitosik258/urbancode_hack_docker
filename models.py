import numpy as np
from ultralytics import YOLO


class YOLOmodel:

    def __init__(self,
                 weights_path: str,
                 cfg_path: str = None):
        
        self.model = YOLO(weights_path)

    def select(self, results, threshold):

        residuals = results

        bboxes = results[:, :-2]

        scores = residuals[:, -2]

        labels = residuals[:, -1] + 1

        mask = (scores > threshold).flatten()

        return bboxes[mask], labels[mask], scores[mask]
    
    def predict(self, image: np.array, threshold: float = .0, device: str = 'cpu'):

        predictions = self.model.predict(image, device=device)

        for pred in predictions:

            residuals = pred.boxes.data.numpy()

        selected = self.select(residuals, threshold)

        return selected


YOLOmodel(weights_path='weights/best.pt')
