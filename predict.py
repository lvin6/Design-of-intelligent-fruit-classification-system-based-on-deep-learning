import torch
import json
from torchvision.transforms import transforms
from fruits_classification_model import FruitsClassificationModel



def predict(img):
    data_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    model = FruitsClassificationModel()
    model.load_state_dict(torch.load("./assets/fruits_classification_model.pth"))  # load trained model
    model.eval()
    classes = json.loads(open("./assets/labels.json").read())

    with torch.no_grad():
        output = torch.squeeze(model(img))
        _, pred = torch.max(output.data, dim=0)
        index = pred.data.numpy()
        result = {k: v for k, v in classes.items() if v == index}
        arr = list(result.keys())
        return arr[0] if len(arr) else "unknown"
