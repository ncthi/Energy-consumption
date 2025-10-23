import argparse
import time
from tqdm import tqdm
import torch
import timm
def main():
    parser = argparse.ArgumentParser(description="Run model inference with timm")
    parser.add_argument("--model", type=str, default="resnet18", help="timm model name to run")
    parser.add_argument("--batch-size", "-b", type=int, default=1, help="batch size for inputs")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = torch.randn((args.batch_size, 3, 224, 224)).to(device)

    model = timm.create_model(args.model, pretrained=False).to(device)
    model.eval()

    with torch.no_grad():
        for _ in range(10):
            model(inputs)
        time.time()
        print("start time:",time.time())
        try:
            for _ in range(500):
                model(inputs)
        except KeyboardInterrupt:
            pass
        print("end time:",time.time())




if __name__ == "__main__":
    main()
