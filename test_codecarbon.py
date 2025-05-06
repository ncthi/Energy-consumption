from codecarbon import track_emissions
import timm
import torch
device="cpu"
input=torch.randn(32,3,224,224).to(device)
@track_emissions()
def your_function_to_track():
    model=timm.create_model('efficientvit_b3', pretrained=False).to(device)
    model.eval()
    model(input)

if __name__=='__main__':
    your_function_to_track()
