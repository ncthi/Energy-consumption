
import torch
import timm
device="cpu"
output=torch.randn(32,3,224,224).to(device)

import torch
import eco2ai

def get_layers(module):
    for layer in module.children():
        if isinstance(layer, torch.nn.Module):
            name = layer.__class__.__name__
            tracker = eco2ai.Tracker(
                project_name="efficientvit",
                experiment_description=name,
                file_name="efficientvit.csv"
            )

            # tạo pre-hook closure
            def make_pre_hook(tracker):
                def _pre_hook(module, inputs):
                    tracker.start()
                return _pre_hook

            # tạo post-hook closure
            def make_post_hook(tracker):
                def _post_hook(module, inputs, output):
                    tracker.stop()
                return _post_hook

            # đăng ký hook
            layer._pre_handle  = layer.register_forward_pre_hook(make_pre_hook(tracker))
            layer._post_handle = layer.register_forward_hook(make_post_hook(tracker))
            get_layers(layer)


if __name__=='__main__':
    model=timm.create_model('efficientvit_b3', pretrained=False).to(device)
    model.eval()
    get_layers(model)
    model(output)


