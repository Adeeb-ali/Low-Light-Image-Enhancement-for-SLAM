import torch

from .core.sampler import ArchitectureSampler

from .core.scorer import ArchitectureScorer

from .core.early_stop import EarlyStopping

from .core.retrain import Retrainer


device = torch.device(

    "cuda"
    if torch.cuda.is_available()
    else "cpu"

)


sampler = ArchitectureSampler()

scorer = ArchitectureScorer(

    device=device

)

early_stop = EarlyStopping(

    patience=3

)


best_result = None


for iteration in range(10):

    print(f"\nNAS Iteration : {iteration + 1}")

    architecture = sampler.sample()

    print(f"Architecture : {architecture}")

    result = scorer.score(

        architecture

    )

    print(f"Loss  : {result['loss']}")

    print(f"Score : {result['score']}")

    if (

        best_result is None

        or

        result["score"] > best_result["score"]

    ):

        best_result = result

        print("New Best Architecture Found")

    stop = early_stop.should_stop(

        result["score"]

    )

    if stop:

        print("Early Stopping Triggered")

        break


print("\nBest Architecture Found")

print(best_result)


retrainer = Retrainer(

    device=device

)

best_model = retrainer.retrain(

    architecture=best_result["architecture"],

    epochs=200

)

print("\nNAS Pipeline Completed")