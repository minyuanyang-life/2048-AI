import torch
from torch import nn
from torch.nn import init


class QNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        init.xavier_uniform_(self.fc3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def main():
    input_size = 16
    output_size = 4
    model = QNetwork(input_size, output_size)

    state = torch.randn(1, 16)

    q_values = model(state)
    print(q_values)

if __name__ == "__main__":
    main()