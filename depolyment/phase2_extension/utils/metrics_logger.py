import os

import csv


class MetricsLogger:

    def __init__(

        self,
        save_dir="outputs/training_logs"

    ):

        self.save_dir = save_dir

        os.makedirs(

            self.save_dir,

            exist_ok=True

        )

        self.csv_path = os.path.join(

            self.save_dir,

            "training_metrics.csv"

        )

        if not os.path.exists(

            self.csv_path

        ):

            with open(

                self.csv_path,

                mode="w",

                newline=""

            ) as file:

                writer = csv.writer(file)

                writer.writerow([

                    "Epoch",

                    "TotalLoss",

                    "Charbonnier",

                    "SSIM",

                    "Edge",

                    "LearningRate"

                ])

    def log(

        self,
        epoch,
        total_loss,
        charbonnier,
        ssim,
        edge,
        learning_rate

    ):

        with open(

            self.csv_path,

            mode="a",

            newline=""

        ) as file:

            writer = csv.writer(file)

            writer.writerow([

                epoch,

                total_loss,

                charbonnier,

                ssim,

                edge,

                learning_rate

            ])

        print(

            f"\nMetrics Saved -> Epoch {epoch}\n"

        )