import torch
import torch.nn as nn


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        print("fc1_output", x.dtype)
        x = self.ln(x)
        print("ln_output", x.dtype)
        x = self.fc2(x)
        print("fc2_output", x.dtype)
        return x


if __name__ == "__main__":
    x = torch.ones(1, 10, dtype=torch.float16).cuda()
    model = ToyModel(10, 1).cuda()

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for name, param in model.named_parameters():
            print(name, param.dtype)
        l = model(x).pow(2)
        print("loss", l.dtype)
        l.backward()
        for name, param in model.named_parameters():
            print(name + ".grad", param.grad.dtype)
