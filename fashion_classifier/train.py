class Learner:
    """
    Learner class for training and evaluating a model.

    It encapsulates the training and validation loops, as well as utility
    methods for prediction, exporting the model, and calculating accuracy.
    """

    def __init__(self, config, loaders):
        """
        Initialize the Learner.

        Args:
            config (LearnerConfig): Configuration for the Learner.
            loaders (dict): Dictionary of data loaders for training and testing.
        """
        self.model = config.model
        self.loaders = loaders
        self.optimizer = BasicOptim(self.model.parameters(), config.lr)
        self.criterion = config.criterion
        self.epochs = config.epochs
        self.device = config.device

    def train_epoch(self, epoch):
        """
        Train the model for one epoch.

        Args:
            epoch (int): Current epoch number.

        Returns:
            float: Average loss for the epoch.
        """
        epoch_loss = 0.0
        for x, y in self.loaders["train"]:
            x, y = x.to(self.device), y.to(self.device)
            batch_size = x.size(0)

            # Zero out the gradients - otherwise, they will accumulate.
            self.optimizer.zero_grad()

            # Forward pass, loss calculation, and backpropagation
            output = self.model(x)
            loss = self.criterion(output, y)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item() * batch_size

        avg_loss = epoch_loss / len(self.loaders["train"].dataset)
        return avg_loss

    def batch_accuracy(self, x, y):
        """
        Calculate the accuracy for a batch of inputs and targets.

        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): Target tensor.

        Returns:
            float: Accuracy for the batch.
        """
        _, preds = torch.max(x.data, 1)
        return (preds == y).sum().item() / x.size(0)

    def validate_epoch(self):
        """
        Evaluate the model on the test dataset.

        Returns:
            float: Average accuracy on the test dataset.
        """
        accs = [
            self.batch_accuracy(self.model(x.to(self.device)), y.to(self.device))
            for x, y in self.loaders["test"]
        ]
        return sum(accs) / len(accs)

    def fit(self):
        """
        Train the model for the specified number of epochs.

        Prints the training loss and test accuracy for each epoch.
        """
        print("epoch\ttraining_loss\ttest_accuracy")
        for epoch in range(self.epochs):
            epoch_loss = self.train_epoch(epoch)
            epoch_accuracy = self.validate_epoch()
            print(f"{epoch+1}\t{epoch_loss:.6f}\t{epoch_accuracy:.6f}")

    def predict(self, x):
        """
        Make predictions on a batch of inputs.
        """
        with torch.no_grad():
            outputs = self.model(x.to(self.device))
            _, predicted = torch.max(outputs.data, 1)
        return predicted

    def predict_probs(self, x):
        """
        Compute the class probabilities for a batch of inputs.
        """
        with torch.no_grad():
            output = self.model(x.to(self.device))
            probabilities = torch.nn.functional.softmax(output, dim=1)
            """
            The softmax function converts a vector of raw model outputs (logits)
            into a probability distribution, wherein the outputs sum 1.
            """
        return probabilities

    def export(self, path):
        """
        Export the model to a file.
        """
        torch.save(self.model, path)
