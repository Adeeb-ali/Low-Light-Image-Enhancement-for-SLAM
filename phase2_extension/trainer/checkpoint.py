import os

import torch


class CheckpointManager:

    def __init__(

        self,
        save_dir

    ):

        self.save_dir = save_dir

        os.makedirs(

            self.save_dir,

            exist_ok=True

        )

    # =========================================================
    # SAVE CHECKPOINT
    # =========================================================

    def save_checkpoint(

        self,
        model,
        optimizer,
        epoch,
        loss,
        channels,
        num_rrdb_blocks,
        filename="latest_model.pth"

    ):

        checkpoint_path = os.path.join(

            self.save_dir,

            filename

        )

        torch.save(

            {

                "epoch": epoch,

                "channels": channels,

                "num_rrdb_blocks": num_rrdb_blocks,

                "model_state_dict": model.state_dict(),

                "optimizer_state_dict": optimizer.state_dict(),

                "loss": loss

            },

            checkpoint_path

        )

        print(

            f"\nCheckpoint Saved : {checkpoint_path}\n"

        )

    # =========================================================
    # LOAD CHECKPOINT
    # =========================================================

    def load_checkpoint(

        self,
        model,
        optimizer,
        checkpoint_path

    ):

        checkpoint = torch.load(

            checkpoint_path,

            map_location="cpu"

        )

        model.load_state_dict(

            checkpoint["model_state_dict"]

        )

        optimizer.load_state_dict(

            checkpoint["optimizer_state_dict"]

        )

        epoch = checkpoint["epoch"]

        loss = checkpoint["loss"]

        print(

            "\n===================================="
        )

        print(

            "CHECKPOINT LOADED"
        )

        print(

            "====================================\n"
        )

        print(

            f"Epoch             : {epoch}"
        )

        print(

            f"Loss              : {loss:.6f}"
        )

        print(

            f"Channels          : {checkpoint['channels']}"
        )

        print(

            f"RRDB Blocks       : {checkpoint['num_rrdb_blocks']}"
        )

        print(

            "\n====================================\n"
        )

        return (

            model,
            optimizer,
            epoch,
            loss

        )