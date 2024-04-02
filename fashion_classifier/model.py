import torch
import torch.nn as nn

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, std: float = 0.1):
        """
        Initialize the linear layer with random values for weights and biases.

        The weights and biases are registered as parameters, allowing for gradient
        computation and update during backpropagation.
        """
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        weight = torch.randn(in_features, out_features, device=device) * std
        bias = torch.randn(out_features, device=device) * std

        self.weight = nn.Parameter(weight.requires_grad_())
        self.bias = nn.Parameter(bias.requires_grad_())
        self.to(device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform linear transformation by multiplying the input tensor
        with the weight matrix and adding the bias.
        """
        return x @ self.weight + self.bias

    def __repr__(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class ReLU(nn.Module):
    """
    Rectified Linear Unit (ReLU) activation function.
    """

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        """
        Apply the ReLU activation function to the input tensor.
        """
        return torch.max(x, torch.tensor(0))


class Sequential(nn.Module):
    """
    Sequential container for stacking multiple modules,
    passing the output of one module as input to the next.
    """

    def __init__(self, *layers):
        """
        Initialize the Sequential container with a list of layers.
        """
        super(Sequential, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the forward pass through the Sequential container.
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self) -> str:
        layer_str = "\n".join(
            [f" ({i}): {layer}" for i, layer in enumerate(self.layers)]
        )
        return f"{self.__class__.__name__}(\n{layer_str}\n)"


class Flatten(nn.Module):
    """
    Reshape the input tensor by flattening all dimensions except the first dimension.
    """

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        """
        Note that x.view(x.size(0), -1) reshapes the x tensor to (x.size(0), N)
        where N is the product of the remaining dimensions.

        E.g. (batch_size, 28, 28) -> (batch_size, 784)
        """
        return x.view(x.size(0), -1)


class Classifier(nn.Module):
    """
    Classifier model consisting of a sequence of linear layers and ReLU activations,
    followed by a final linear layer that outputs logits (unnormalized scores)
    for each of the 10 garment classes.
    """

    def __init__(self):
        """
        Note that the 10 output logits of the last layer can be passed directly
        to a loss function like nn.CrossEntropyLoss, which applies the softmax
        function internally to calculate a probability distribution.
        """
        super(Classifier, self).__init__()
        self.main = Sequential(
            Flatten(),
            Linear(in_features=784, out_features=512),
            ReLU(),
            Linear(in_features=512, out_features=256),
            ReLU(),
            Linear(in_features=256, out_features=64),
            ReLU(),
            Linear(in_features=64, out_features=10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass through the classifier model.
        """
        return self.main(x)
