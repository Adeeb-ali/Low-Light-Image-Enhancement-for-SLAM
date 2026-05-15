class EarlyStopping:

    def __init__(

        self,

        patience=3

    ):

        self.patience = patience

        self.best_score = None

        self.counter = 0

    def should_stop(

        self,

        current_score

    ):

        if self.best_score is None:

            self.best_score = current_score

            return False

        if current_score > self.best_score:

            self.best_score = current_score

            self.counter = 0

            return False

        self.counter += 1

        if self.counter >= self.patience:

            return True

        return False