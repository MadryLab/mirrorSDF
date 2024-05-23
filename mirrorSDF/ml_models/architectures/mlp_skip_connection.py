import torch as ch


class MLPWithSkip(ch.nn.Module):
    """
       A Multilayer Perceptron (MLP) module with configurable depth, width, and optional skip connections.

       This class also uses the Kaiming uniform initialization

       Parameters
       ----------
       input_features : int
           The number of input features.
       output_features : int
           The number of output features.
       width : int, optional
           The width of the network, i.e., the number of neurons in each hidden layer. Default is 256.
       n_layers : int, optional
           The total number of layers in the network, including the output layer. Default is 3.
       activation : torch.nn.Module, optional
           The activation function to use after each layer, except for the output layer. Default is torch.nn.ReLU.
       skip_connections : tuple of int, optional
           A tuple specifying the zero-indexed positions of layers after which skip connections should be added.
           For instance, (4,) means add a skip connection after the 4th layer. Default is (4,).

       Attributes
       ----------
       activation : torch.nn.Module
           The activation function module.
       layers : torch.nn.ModuleList
           A ModuleList containing all layers of the MLP, including both hidden and output layers.
       last_layer : torch.nn.Linear
           A direct reference to the last layer in the network for potential customization or inspection.
       """
    def __init__(self, input_features: int, output_features: int,
                 width: int = 256, n_layers: int = 3,
                 activation=ch.nn.ReLU, skip_connections=(4, 8, 12)):
        super().__init__()

        # Ensure we have at least one layer
        assert n_layers > 0, "Number of layers should be greater than 0"

        self.activation = activation()

        # Initial layers list
        layers = [ch.nn.Linear(input_features, width)]

        # Hidden layers
        for cur_layer in range(n_layers - 1):  # -1 as we've already added the input layer
            cur_inputs = width
            if cur_layer in skip_connections:
                cur_inputs += input_features

            layers.append(ch.nn.Linear(cur_inputs, width))

        layers.append(ch.nn.Linear(width, output_features))

        self.layers = ch.nn.ModuleList(layers)
        self.last_layer: ch.nn.Linear = self.layers[-1]

        for p in self.parameters():
            if len(p.shape) == 2:
                ch.nn.init.kaiming_uniform_(p, nonlinearity='relu')

    def forward(self, x):
        """
        Forward pass through the MLP with optional skip connections.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor after passing through the MLP.
        """
        original_input = x

        for i, layer in enumerate(self.layers):
            if layer.in_features > x.shape[-1]:
                x = ch.cat([original_input, x], dim=-1)
            x = layer(x)
            if i != len(self.layers) - 1:
                x = self.activation(x)

        return x
