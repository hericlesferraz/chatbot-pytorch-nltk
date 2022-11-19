
class ChatDataset:
    def __init__(self, x_train, y_train) -> None:
        self.num_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]

    def __len__(self):
        return self.num_samples
