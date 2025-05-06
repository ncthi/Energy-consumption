from carbontracker.tracker import CarbonTracker
import timm
import torch
max_epochs=1

device="cpu"
input=torch.randn(32,3,224,224).to(device)
tracker = CarbonTracker(epochs=max_epochs)

# Training loop.
for epoch in range(max_epochs):
    model=timm.create_model('efficientvit_b3', pretrained=False).to(device)
    tracker.epoch_start()
    model.eval()
    model(input)
    tracker.epoch_end()
tracker.stop()