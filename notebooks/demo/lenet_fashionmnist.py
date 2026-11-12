import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dissociating representation from function and behaviour

    A minimal, runnable illustration of the central result of
    *"Representation can dissociate from function and behaviour in
    task-optimised vision networks"* (CCN 2026): a network can be trained
    to achieve a chosen representational similarity to a reference network
    while its **task performance, output logits and trial-by-trial
    behaviour stay indistinguishable from those of ordinary, independently
    trained networks**.

    The abstract's main experiment sweeps the target similarity of
    ResNet-50 on ImageNet (Fig. 1A). This notebook demonstrates the
    LeNet-5 / Fashion-MNIST experiment with a target linear CKA of 0.0.
    Optimisation aims to reduce representational similarity while matching
    the reference network's output logits. The achieved CKA, accuracy and
    error consistency are evaluated below.

    **Procedure:**

    1. Load Fashion-MNIST and train a small LeNet-5 as the fixed
       **reference model** (or load the bundled checkpoint).
    2. Build a second, initially identical **train model**, freeze its
       convolutional feature extractor and re-initialise its classifier
       head.
    3. Train only the classifier head with an **augmented Lagrangian loss**
       that minimises the output MSE to the reference (the *functional*
       objective) while *constraining* the linear CKA at the penultimate
       layer to the target score, rather than simply minimising or
       maximising it.
    4. Compare the two networks on logits (function), error consistency
       (behaviour), weight distributions and layer-wise CKA
       (representation), the quantities Fig. 1 reports.
    """)
    return


@app.cell
def _(mo):
    import random
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.data import Subset
    from torchvision import datasets
    from torchvision import transforms

    from dissociation import colors
    from dissociation import plotting
    from dissociation import results
    from dissociation.figure import MEAN_LINE_DASHES
    from dissociation.losses import AugmentedLagrangianLoss
    from dissociation.losses import MSERef
    from dissociation.metrics import error_consistency_kappa
    from dissociation.models import LeNet
    from dissociation.models import freeze_model_layers
    from dissociation.models import get_layer_names
    from dissociation.models import randomly_initialize_layers
    from dissociation.similarity import CKA

    # Paths are resolved relative to the repository, whether the notebook
    # runs in `marimo edit` or as a plain script.
    _notebook_dir = mo.notebook_dir()
    REPO_ROOT = (
        _notebook_dir.parents[1] if _notebook_dir else Path.cwd().resolve()
    )
    DATA_DIR = REPO_ROOT / "data" / "fashionmnist"  # downloaded on demand
    OUT_DIR = REPO_ROOT / "out" / "demo"  # checkpoints written by this demo
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return (
        AugmentedLagrangianLoss,
        CKA,
        DATA_DIR,
        DataLoader,
        LeNet,
        MEAN_LINE_DASHES,
        MSERef,
        OUT_DIR,
        REPO_ROOT,
        Subset,
        colors,
        datasets,
        error_consistency_kappa,
        freeze_model_layers,
        get_layer_names,
        np,
        plotting,
        plt,
        random,
        randomly_initialize_layers,
        results,
        torch,
        transforms,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Configuration

    LeNet-5 on Fashion-MNIST is the smallest network/dataset pairing in the
    abstract (alongside AlexNet / CIFAR-10 and VGG, ResNet and ViT on
    ImageNet). This configuration supports short training runs on a laptop.
    The split is deterministic and data loading stays in the main process.
    """)
    return


@app.cell
def _(np, random, torch):
    SEED = 2026
    BATCH_SIZE = 128
    NUM_WORKERS = 0
    VAL_SIZE = 10000

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Device:", device)
    return BATCH_SIZE, NUM_WORKERS, SEED, VAL_SIZE, device


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Data

    LeNet-5 expects 32 x 32 grayscale inputs (Fashion-MNIST images are
    28 x 28). Training uses a random resized crop and random erasing for
    regularisation; validation and test use a deterministic resize only, so
    none of the evaluation numbers below depend on augmentation randomness.
    """)
    return


@app.cell
def _(
    BATCH_SIZE,
    DATA_DIR,
    DataLoader,
    NUM_WORKERS,
    SEED,
    Subset,
    VAL_SIZE,
    datasets,
    torch,
    transforms,
):
    # Augmentation parameters come from an earlier hyperparameter search.
    _mean = (0.2860,)
    _std = (0.3205,)

    _train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=(32, 32),
                scale=(0.9318, 1.0),
                ratio=(0.7693, 1.2999),
            ),
            transforms.ToTensor(),
            transforms.Normalize(_mean, _std),
            transforms.RandomErasing(
                p=0.0796,
                scale=(0.1164, 0.2778),
                ratio=(0.3688, 2.7117),
            ),
        ]
    )
    _eval_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(_mean, _std),
        ]
    )

    _train_full = datasets.FashionMNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=_train_transform,
    )
    _val_full = datasets.FashionMNIST(
        root=DATA_DIR,
        train=True,
        download=False,
        transform=_eval_transform,
    )
    _test_data = datasets.FashionMNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=_eval_transform,
    )

    # Split the 60k training images into train and validation subsets.
    _train_size = len(_train_full) - VAL_SIZE
    _generator = torch.Generator().manual_seed(SEED)
    _indices = torch.randperm(len(_train_full), generator=_generator).tolist()
    _train_data = Subset(_train_full, _indices[:_train_size])
    _val_data = Subset(_val_full, _indices[_train_size:])

    _loader_kwargs = {"batch_size": BATCH_SIZE, "num_workers": NUM_WORKERS}
    train_loader = DataLoader(_train_data, shuffle=True, **_loader_kwargs)
    val_loader = DataLoader(_val_data, shuffle=False, **_loader_kwargs)
    test_loader = DataLoader(_test_data, shuffle=False, **_loader_kwargs)

    class_names = _train_full.classes
    print(
        "Dataset sizes (train / val / test):",
        len(_train_data),
        len(_val_data),
        len(_test_data),
    )
    print("Classes:", ", ".join(class_names))
    return class_names, test_loader, train_loader, val_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reference model

    The fixed **reference (target) model** everything below is compared
    against; it plays the role of "the target model" in the abstract. It is
    trained normally: cross-entropy, SGD and a validation-loss scheduler.
    By default the bundled checkpoint `models/lenet_fashionmnist_reference.pt`
    is loaded (`LOAD_REF_CHECKPOINT = True`); set it to `False` to train
    your own, which is then written to `out/demo/` and used instead.
    """)
    return


@app.cell
def _(OUT_DIR, REPO_ROOT):
    N_EPOCHS = 80
    LR = 0.1
    MOMENTUM = 0.6
    WEIGHT_DECAY = 1e-4
    SCHEDULER_FACTOR = 0.4
    SCHEDULER_PATIENCE = 8
    EARLY_STOP_PATIENCE = 10

    LOAD_REF_CHECKPOINT = True
    REF_CHECKPOINT_PATH = (
        REPO_ROOT / "models" / "lenet_fashionmnist_reference.pt"
        if LOAD_REF_CHECKPOINT
        else OUT_DIR / "lenet_fashionmnist_reference.pt"
    )
    return (
        EARLY_STOP_PATIENCE,
        LOAD_REF_CHECKPOINT,
        LR,
        MOMENTUM,
        N_EPOCHS,
        REF_CHECKPOINT_PATH,
        SCHEDULER_FACTOR,
        SCHEDULER_PATIENCE,
        WEIGHT_DECAY,
    )


@app.cell
def _(device, torch):
    @torch.no_grad()
    def evaluate_classifier(model, loader, criterion):
        """Return (mean loss, accuracy) of a classifier over a loader."""
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for batch_images, batch_labels in loader:
            images = batch_images.to(device)
            labels = batch_labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size
        return total_loss / total_samples, total_correct / total_samples

    return (evaluate_classifier,)


@app.cell
def _(
    EARLY_STOP_PATIENCE,
    LOAD_REF_CHECKPOINT,
    LR,
    LeNet,
    MOMENTUM,
    N_EPOCHS,
    REF_CHECKPOINT_PATH,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    WEIGHT_DECAY,
    device,
    evaluate_classifier,
    torch,
    train_loader,
    val_loader,
):
    _model = LeNet().to(device)
    _criterion = torch.nn.CrossEntropyLoss()

    if LOAD_REF_CHECKPOINT and REF_CHECKPOINT_PATH.exists():
        _checkpoint = torch.load(
            REF_CHECKPOINT_PATH, map_location=device, weights_only=True
        )
        _model.load_state_dict(_checkpoint["model_state"])
        _val_loss, _val_acc = evaluate_classifier(
            _model, val_loader, _criterion
        )
        print(f"Loaded reference checkpoint {REF_CHECKPOINT_PATH.name}.")
        print(f"Validation: loss {_val_loss:.3f}, accuracy {_val_acc:.3f}")
    else:
        _optimizer = torch.optim.SGD(
            _model.parameters(),
            lr=LR,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
        )
        _scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            _optimizer,
            mode="min",
            factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE,
        )
        _best_val_acc = float("-inf")
        _stale_epochs = 0
        print("Training the reference model.")
        for _epoch in range(1, N_EPOCHS + 1):
            _model.train()
            _train_loss = 0.0
            _train_correct = 0
            _train_samples = 0
            for _images, _labels in train_loader:
                _images = _images.to(device)
                _labels = _labels.to(device)
                _optimizer.zero_grad(set_to_none=True)
                _logits = _model(_images)
                _loss = _criterion(_logits, _labels)
                _loss.backward()
                _optimizer.step()
                _batch_size = _labels.size(0)
                _train_loss += _loss.item() * _batch_size
                _train_correct += (
                    (_logits.argmax(dim=1) == _labels).sum().item()
                )
                _train_samples += _batch_size
            _train_loss /= _train_samples
            _train_acc = _train_correct / _train_samples
            _val_loss, _val_acc = evaluate_classifier(
                _model, val_loader, _criterion
            )
            _scheduler.step(_val_loss)
            print(
                f"Epoch {_epoch:02d}/{N_EPOCHS} | "
                f"train: loss {_train_loss:.3f}, acc {_train_acc:.3f} | "
                f"val: loss {_val_loss:.3f}, acc {_val_acc:.3f}"
            )
            if _val_acc > _best_val_acc:
                _best_val_acc = _val_acc
                torch.save(
                    {
                        "epoch": _epoch,
                        "val_loss": _val_loss,
                        "val_acc": _val_acc,
                        "model_state": {
                            key: value.detach().cpu()
                            for key, value in _model.state_dict().items()
                        },
                    },
                    REF_CHECKPOINT_PATH,
                )
                _stale_epochs = 0
            else:
                _stale_epochs += 1
            if _stale_epochs >= EARLY_STOP_PATIENCE:
                print(
                    f"Early stopping at epoch {_epoch:02d}/{N_EPOCHS}; "
                    f"best validation accuracy {_best_val_acc:.3f} at epoch "
                    f"{_epoch - _stale_epochs:02d}."
                )
                break
            if _epoch == N_EPOCHS:
                print(
                    f"Reached the last epoch; best validation accuracy "
                    f"{_best_val_acc:.3f}."
                )

    # Later cells read the checkpoint, so they depend on this cell.
    reference_checkpoint = REF_CHECKPOINT_PATH
    return (reference_checkpoint,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Train model setup

    The **train model** starts as an exact copy of the reference (same
    weights), then:

    - **Freezes `net.0` to `net.8`**, the entire convolutional feature
      extractor (conv/pool/ReLU stack up to and including the flatten).
      These layers stay identical to the reference throughout training.
      This is the "frozen feature extractor" of the abstract's methods:
      fixing early visual features means any representational change we
      induce has to originate from the classifier head.
    - **Re-initialises `net.9` to `net.11`**, the classifier head: `net.9`
      (`Linear(120, 84)`), `net.10` (`ReLU`) and `net.11`
      (`Linear(84, 10)`, the output layer). Only `net.9` and `net.11` have
      weights to reset; `net.10` is a parameter-free ReLU, included only
      because the helpers select layers by *index range*.
    - **Hooks `net.10`**, the activation right after the first classifier
      layer and the last representation before the final linear readout.
      This is the **penultimate-layer representation** the similarity
      scores are computed on. The hook stores this activation for both
      models on every forward pass so the CKA constraint can be evaluated.
    """)
    return


@app.cell
def _(
    LeNet,
    device,
    freeze_model_layers,
    get_layer_names,
    randomly_initialize_layers,
    reference_checkpoint,
    torch,
):
    HOOK_LAYER_NAME = "net.10"
    REINIT_LAYERS = "last:3"
    FREEZE_LAYERS = "all_but_last:3"

    _checkpoint = torch.load(
        reference_checkpoint,
        map_location=device,
        weights_only=True,
    )
    model_ref = LeNet().to(device)
    model_ref.load_state_dict(_checkpoint["model_state"])
    model_ref.eval()
    for _param in model_ref.parameters():
        _param.requires_grad = False

    model_train = LeNet().to(device)
    model_train.load_state_dict(_checkpoint["model_state"])
    freeze_model_layers(model_train, FREEZE_LAYERS)
    randomly_initialize_layers(model_train, REINIT_LAYERS)
    model_train.train()

    # Forward hooks store the hooked layer's activation of each model here.
    activations = {"train": {}, "ref": {}}

    def _save_activation(model_key):
        def hook(_module, _inputs, output):
            activation = output.flatten(start_dim=1)
            if model_key == "ref":
                # Gradients only ever flow through the train model.
                activation = activation.detach()
            activations[model_key][HOOK_LAYER_NAME] = activation

        return hook

    _hook_layer_idx = int(HOOK_LAYER_NAME.removeprefix("net."))
    _handles = [
        model_train.net[_hook_layer_idx].register_forward_hook(
            _save_activation("train")
        ),
        model_ref.net[_hook_layer_idx].register_forward_hook(
            _save_activation("ref")
        ),
    ]

    _trainable = [
        name
        for name, param in model_train.named_parameters()
        if param.requires_grad
    ]
    print("Train model ready.")
    print(f"  Hook layer: {HOOK_LAYER_NAME}")
    _reinitialised = get_layer_names(model_train, REINIT_LAYERS)
    print(f"  Re-initialised: {', '.join(_reinitialised)}")
    print(f"  Trainable parameters: {', '.join(_trainable)}")
    return HOOK_LAYER_NAME, activations, model_ref, model_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Augmented Lagrangian training

    Training combines a logit-matching objective with a representational
    similarity constraint:

    1. **Match the reference's outputs** (functional similarity): minimise
       the MSE between train and reference logits.
    2. **Constrain CKA to the target score** at the penultimate layer
       (`net.10`) against the reference by penalising deviations from the
       specified value (here TARGET_SCORE = 0.0).

    For a mini-batch $B$, let
    $s_B(\theta) = \mathrm{CKA}(A_{\theta,B}, A_{\text{ref},B})$
    be the achieved CKA and $c^{*}$ the target. The batch constraint
    residual is $g_B(\theta) = s_B(\theta) - c^{*}$, and the loss
    backpropagated at each step is

    $$
    \mathcal{L}_{\text{AL},B}(\theta) =
    \operatorname{MSE}\big(z_{\theta,B},\, z_{\text{ref},B}\big) +
    \lambda\, g_B(\theta) +
    \frac{\mu}{2}\, g_B(\theta)^2 .
    $$

    A plain weighted sum

    $$
    \mathcal{L}_{\text{WS},B}(\theta) =
    \operatorname{MSE}\big(z_{\theta,B},\, z_{\text{ref},B}\big) +
    \eta\, g_B(\theta)^2
    $$

    requires selecting a fixed weight $\eta$, which may need tuning across
    targets. The augmented Lagrangian instead updates $\lambda$ and $\mu$
    once per epoch, while still requiring choices of update parameters.
    The update uses the mean signed training residual

    $$
    \bar g_k = \frac{1}{M}\sum_{b=1}^{M}g_{B_b}(\theta_{k,b}),
    $$

    where $\theta_{k,b}$ denotes the weights before batch $b$'s update in
    epoch $k$. This averages residuals collected as training proceeds;
    it is not a fresh evaluation at the epoch's final weights.
    `loss_fn.update_parameters` applies the following rule:

    - If $|\bar g_k|$ is no larger than `AL_UPDATE_TOLERANCE` times the
      best accepted value of $|\bar g|$, take a **dual step**:
      $\lambda \leftarrow \lambda + \mu\,\bar g_k$, and record
      $|\bar g_k|$ as the new best value.
    - Otherwise **grow the penalty**:
      $\mu \leftarrow \min(\gamma \mu,\, \mu_{\max})$ with
      $\gamma$ = `AL_MU_UPDATE_FACTOR`.

    The best value is initialised to infinity, so the first epoch accepts
    a dual step. With tolerance 1.0, subsequent dual steps require matching
    or improving the best accepted value, rather than the previous epoch.
    A smaller tolerance requires a larger relative improvement for a dual
    step. The penalty is capped at $\mu_{\max}$.

    CKA is computed per mini-batch; its mean generally differs from CKA
    computed over the full dataset. The loss penalises squared batch
    residuals, whereas the outer update checks the absolute mean signed
    residual. For nonzero targets, positive and negative batch residuals
    can cancel in this mean even when the batch penalties are nonzero.

    This is an adaptation of Dener et al. (2020), using training-epoch
    residual averages in place of constraint evaluations at the completed
    inner-loop solution. The demo reshuffles batches each epoch and retains
    learning-rate and penalty state across epochs, without the paper's
    resets. Checkpoints are selected by validation MSE plus absolute batch
    residual, averaged over batches, rather than feasibility tolerances.

    The augmented Lagrangian updates aim to reduce the difference between
    the achieved and target CKA, while the MSE term penalises differences
    between the train and reference logits.

    The following cells perform one training run at a single target score,
    starting from the reference checkpoint. Change `TARGET_SCORE` (e.g.
    to 0.5 or 1.0) and re-run this and the following cells to see a different
    point of a Fig. 1A-style sweep. A single run (80 epochs) took 7 minutes on
    an Apple M3 Max laptop with the MPS backend and 15 minutes on its CPU cores
    alone (details in the README).
    """)
    return


@app.cell
def _(LR, N_EPOCHS, OUT_DIR):
    TARGET_SCORE = 0.0

    N_AL_EPOCHS = N_EPOCHS
    AL_LR = LR / 25  # only the classifier head is trained
    AL_INITIAL_LAMBDA = 0.0
    AL_INITIAL_MU = 1.5
    AL_MU_UPDATE_FACTOR = 1.2
    AL_MU_MAX = 5000
    AL_UPDATE_TOLERANCE = 1.0  # accept if no worse than the best accepted value

    # The checkpoint name carries the target, so changing TARGET_SCORE never
    # silently reuses a run for a different target.
    TRAIN_CHECKPOINT_PATH = OUT_DIR / f"lenet_cka_target_{TARGET_SCORE:.2f}.pt"
    LOAD_TRAIN_CHECKPOINT = True  # reuse an existing run for this target
    return (
        AL_INITIAL_LAMBDA,
        AL_INITIAL_MU,
        AL_LR,
        AL_MU_MAX,
        AL_MU_UPDATE_FACTOR,
        AL_UPDATE_TOLERANCE,
        LOAD_TRAIN_CHECKPOINT,
        N_AL_EPOCHS,
        TARGET_SCORE,
        TRAIN_CHECKPOINT_PATH,
    )


@app.cell
def _(device, error_consistency_kappa, torch):
    # Two "loss" values are tracked:
    #   aug_loss      -> the augmented Lagrangian objective used for backprop
    #   combined_loss -> MSE + |violation|, an unweighted score used only to
    #                    pick the best checkpoint and for early stopping
    @torch.no_grad()
    def evaluate_al_model(model_train, model_ref, loss_fn, loader, activations):
        """Return the mean metrics of the train model over a loader."""
        model_train.eval()
        model_ref.eval()
        totals = {
            "aug_loss": 0.0,
            "combined_loss": 0.0,
            "mse": 0.0,
            "cka": 0.0,
            "acc": 0.0,
            "ec": 0.0,
        }
        n_batches = 0
        for batch_images, batch_labels in loader:
            activations["train"].clear()
            activations["ref"].clear()
            images = batch_images.to(device)
            labels = batch_labels.to(device)
            train_logits = model_train(images)
            ref_logits = model_ref(images)
            constraint, mse_loss, combined_loss, aug_loss = loss_fn(
                activations, {"train": train_logits, "ref": ref_logits}
            )
            totals["aug_loss"] += aug_loss.item()
            totals["combined_loss"] += combined_loss.item()
            totals["mse"] += mse_loss.item()
            # constraint = CKA - target, so add the target back
            totals["cka"] += (constraint + loss_fn.target_score).item()
            totals["acc"] += (
                (train_logits.argmax(dim=1) == labels).float().mean().item()
            )
            totals["ec"] += error_consistency_kappa(
                train_logits, ref_logits, labels
            ).item()
            n_batches += 1
        model_train.train()
        return {key: value / n_batches for key, value in totals.items()}

    return (evaluate_al_model,)


@app.cell
def _(
    AL_INITIAL_LAMBDA,
    AL_INITIAL_MU,
    AL_LR,
    AL_MU_MAX,
    AL_MU_UPDATE_FACTOR,
    AL_UPDATE_TOLERANCE,
    AugmentedLagrangianLoss,
    CKA,
    HOOK_LAYER_NAME,
    MOMENTUM,
    MSERef,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    TARGET_SCORE,
    WEIGHT_DECAY,
    device,
    model_train,
    torch,
):
    loss_fn = AugmentedLagrangianLoss(
        similarity_measure=CKA(),
        functional_measure=MSERef(),
        target_score=TARGET_SCORE,
        layer_name=HOOK_LAYER_NAME,
        initial_lambda=AL_INITIAL_LAMBDA,
        initial_mu=AL_INITIAL_MU,
        mu_update_factor=AL_MU_UPDATE_FACTOR,
        mu_max=AL_MU_MAX,
        update_tolerance=AL_UPDATE_TOLERANCE,
    ).to(device)
    optimizer_al = torch.optim.SGD(
        (param for param in model_train.parameters() if param.requires_grad),
        lr=AL_LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler_al = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_al,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
    )
    return loss_fn, optimizer_al, scheduler_al


@app.cell
def _(
    EARLY_STOP_PATIENCE,
    LOAD_TRAIN_CHECKPOINT,
    N_AL_EPOCHS,
    TARGET_SCORE,
    TRAIN_CHECKPOINT_PATH,
    activations,
    device,
    evaluate_al_model,
    loss_fn,
    model_ref,
    model_train,
    optimizer_al,
    scheduler_al,
    torch,
    train_loader,
    val_loader,
):
    if LOAD_TRAIN_CHECKPOINT and TRAIN_CHECKPOINT_PATH.exists():
        _checkpoint = torch.load(
            TRAIN_CHECKPOINT_PATH,
            map_location=device,
            weights_only=True,
        )
        model_train.load_state_dict(_checkpoint["model_state"])
        loss_fn.load_state_dict(_checkpoint["loss_state"])
        print(
            f"Loaded train checkpoint {TRAIN_CHECKPOINT_PATH.name} "
            f"(epoch {_checkpoint['epoch']}, validation combined loss "
            f"{_checkpoint['val_metrics']['combined_loss']:.3f})."
        )
    else:
        _best_val_loss = float("inf")
        _stale_epochs = 0
        print(f"Target CKA {TARGET_SCORE:.2f}; training the classifier head.")
        for _epoch in range(1, N_AL_EPOCHS + 1):
            model_train.train()
            _constraints = []
            _totals = {"mse": 0.0, "cka": 0.0, "combined": 0.0, "acc": 0.0}
            _n_batches = 0
            for _images, _labels in train_loader:
                activations["train"].clear()
                activations["ref"].clear()
                _images = _images.to(device)
                _labels = _labels.to(device)
                optimizer_al.zero_grad(set_to_none=True)
                _train_logits = model_train(_images)
                with torch.no_grad():
                    _ref_logits = model_ref(_images)
                _constraint, _mse, _combined, _aug = loss_fn(
                    activations, {"train": _train_logits, "ref": _ref_logits}
                )
                _aug.backward()
                optimizer_al.step()
                _constraints.append(_constraint.detach())
                _totals["mse"] += _mse.item()
                _totals["cka"] += (_constraint + loss_fn.target_score).item()
                _totals["combined"] += _combined.item()
                _totals["acc"] += (
                    (_train_logits.argmax(dim=1) == _labels)
                    .float()
                    .mean()
                    .item()
                )
                _n_batches += 1
            # Update lambda/mu using the mean signed training residual,
            # compared with the best accepted absolute mean residual.
            _lambda, _mu = loss_fn.update_parameters(
                torch.stack(_constraints).mean()
            )
            _val = evaluate_al_model(
                model_train, model_ref, loss_fn, val_loader, activations
            )
            scheduler_al.step(_val["aug_loss"])
            _train = {key: value / _n_batches for key, value in _totals.items()}
            print(
                f"Epoch {_epoch:02d}/{N_AL_EPOCHS} | "
                f"train: mse {_train['mse']:.3f}, cka {_train['cka']:.3f}, "
                f"acc {_train['acc']:.3f} | "
                f"val: mse {_val['mse']:.3f}, cka {_val['cka']:.3f}, "
                f"combined {_val['combined_loss']:.3f}, "
                f"acc {_val['acc']:.3f} | "
                f"lambda {_lambda:.3f}, mu {_mu:.1f}"
            )
            if _val["combined_loss"] < _best_val_loss:
                _best_val_loss = _val["combined_loss"]
                torch.save(
                    {
                        "target_score": TARGET_SCORE,
                        "epoch": _epoch,
                        "model_state": {
                            key: value.detach().cpu()
                            for key, value in model_train.state_dict().items()
                        },
                        "loss_state": {
                            key: value.detach().cpu()
                            for key, value in loss_fn.state_dict().items()
                        },
                        "val_metrics": _val,
                    },
                    TRAIN_CHECKPOINT_PATH,
                )
                _stale_epochs = 0
            else:
                _stale_epochs += 1
            if _stale_epochs >= EARLY_STOP_PATIENCE:
                print(
                    f"Early stopping at epoch {_epoch:02d}/{N_AL_EPOCHS}; "
                    f"best validation combined loss {_best_val_loss:.3f}."
                )
                break
            if _epoch == N_AL_EPOCHS:
                print(
                    f"Reached the last epoch; best validation combined loss "
                    f"{_best_val_loss:.3f}."
                )

    train_checkpoint = TRAIN_CHECKPOINT_PATH
    return (train_checkpoint,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Test-set summary

    Evaluate the best checkpoint (selected by the validation
    `combined_loss`) on the held-out test set. This run evaluates one
    target CKA score, corresponding to one point in a sweep such as
    illustrated in Fig. 1A. `cka` should approximate `TARGET_SCORE`,
    while `acc` and `ec` (error consistency with the reference) should
    be comparable to those of ordinary, independently trained LeNet
    models. With the default settings the run ends at a CKA around 0.07.
    With linear final readouts, exact zero CKA and exact agreement on
    nonconstant logits are incompatible on the same batch. The zero target
    is therefore a requested value; the reported results are approximate.
    """)
    return


@app.cell
def _(
    AL_INITIAL_LAMBDA,
    AL_INITIAL_MU,
    AL_MU_MAX,
    AL_MU_UPDATE_FACTOR,
    AL_UPDATE_TOLERANCE,
    AugmentedLagrangianLoss,
    CKA,
    HOOK_LAYER_NAME,
    MSERef,
    TARGET_SCORE,
    activations,
    device,
    evaluate_al_model,
    model_ref,
    model_train,
    test_loader,
    torch,
    train_checkpoint,
):
    _checkpoint = torch.load(
        train_checkpoint,
        map_location=device,
        weights_only=True,
    )
    model_train.load_state_dict(_checkpoint["model_state"])
    model_train.eval()
    # A fresh loss object with the checkpoint's lambda/mu, so the evaluation
    # does not depend on how far training went on after the best epoch.
    _loss_fn = AugmentedLagrangianLoss(
        similarity_measure=CKA(),
        functional_measure=MSERef(),
        target_score=TARGET_SCORE,
        layer_name=HOOK_LAYER_NAME,
        initial_lambda=AL_INITIAL_LAMBDA,
        initial_mu=AL_INITIAL_MU,
        mu_update_factor=AL_MU_UPDATE_FACTOR,
        mu_max=AL_MU_MAX,
        update_tolerance=AL_UPDATE_TOLERANCE,
    ).to(device)
    _loss_fn.load_state_dict(_checkpoint["loss_state"])
    test_metrics = {
        "target_score": TARGET_SCORE,
        **evaluate_al_model(
            model_train, model_ref, _loss_fn, test_loader, activations
        ),
    }
    print(
        f"Test: mse {test_metrics['mse']:.3f}, "
        f"error consistency {test_metrics['ec']:.3f}, "
        f"cka {test_metrics['cka']:.3f}, "
        f"acc {test_metrics['acc']:.3f}."
    )
    return (test_metrics,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Trained vs reference

    Four views of the same comparison, covering the quantities Fig. 1
    reports:

    - **Mean logits per class** (top left): functional similarity, for the
      selected true class.
    - **Error-consistency table** (top right): behavioural similarity, i.e.
      how often the two models are *right* or *wrong* on the same test
      images beyond what their accuracies predict (Cohen's kappa; Geirhos
      et al., 2020).
    - **Split weight distributions** (bottom left): a comparison of the
      trained and reference classifier heads' weight distributions at
      `net.9` and `net.11`.
    - **Layer-wise CKA** (bottom right): representational similarity.
      "Early layers" (`net.0` to `net.8`) are trivially at 1.0 since that
      backbone is frozen and shared with the reference; `net.9` and
      `net.11` show whether the imposed dissimilarity at the constrained
      layer (`net.10`) carries over to the neighbouring classifier layers,
      which were only *indirectly* optimised by the MSE and CKA gradients.

    Use the dropdown to inspect different true classes in the top-left
    panel.

    **Reference.** Geirhos, R., Meding, K., & Wichmann, F. A. (2020). Beyond
    accuracy: Quantifying trial-by-trial behaviour of CNNs and humans by
    measuring error consistency. In H. Larochelle, M. Ranzato, R. Hadsell,
    M. F. Balcan, & H. Lin (Eds.), _Advances in Neural Information Processing
    Systems_ (Vol. 33, pp. 13890–13902). Curran Associates.
    https://proceedings.neurips.cc/paper/2020/hash/9f6992966d4c363ea0162a056cb45fe5-Abstract.html
    """)
    return


@app.cell
def _(class_names, mo):
    selected_true_class = mo.ui.dropdown(
        options=class_names,
        value=class_names[0],
        label="True class",
    )
    return (selected_true_class,)


@app.cell
def _(
    CKA,
    device,
    error_consistency_kappa,
    model_ref,
    model_train,
    np,
    test_loader,
    test_metrics,
    torch,
):
    _ = test_metrics  # run after the test cell has loaded the best weights

    @torch.no_grad()
    def _collect_analysis_data():
        # The constrained layer (net.10) plus its trainable neighbours.
        layer_names = ("net.9", "net.10", "net.11")
        layer_activations = {"train": {}, "ref": {}}
        cka = CKA().to(device)
        cka_values = {layer_name: [] for layer_name in layer_names}
        train_logits, ref_logits, labels, handles = [], [], [], []

        def save_activation(model_key, layer_name):
            def hook(_module, _inputs, output):
                layer_activations[model_key][layer_name] = (
                    output.detach().flatten(start_dim=1)
                )

            return hook

        try:
            for layer_name in layer_names:
                idx = int(layer_name.removeprefix("net."))
                handles.append(
                    model_train.net[idx].register_forward_hook(
                        save_activation("train", layer_name)
                    )
                )
                handles.append(
                    model_ref.net[idx].register_forward_hook(
                        save_activation("ref", layer_name)
                    )
                )
            model_train.eval()
            model_ref.eval()
            for batch_images, batch_labels in test_loader:
                layer_activations["train"].clear()
                layer_activations["ref"].clear()
                images = batch_images.to(device)
                train_logits.append(model_train(images).cpu())
                ref_logits.append(model_ref(images).cpu())
                labels.append(batch_labels)
                for layer_name in layer_names:
                    cka_values[layer_name].append(
                        cka(
                            layer_activations["train"][layer_name],
                            layer_activations["ref"][layer_name],
                        ).item()
                    )
        finally:
            for handle in handles:
                handle.remove()

        train_logits = torch.cat(train_logits)
        ref_logits = torch.cat(ref_logits)
        labels = torch.cat(labels)
        train_correct = train_logits.argmax(dim=1) == labels
        ref_correct = ref_logits.argmax(dim=1) == labels
        error_matrix = torch.tensor(
            [
                [
                    (ref_correct & train_correct).sum(),
                    (ref_correct & ~train_correct).sum(),
                ],
                [
                    (~ref_correct & train_correct).sum(),
                    (~ref_correct & ~train_correct).sum(),
                ],
            ],
            dtype=torch.float32,
        )

        def weights(model, idx):
            return model.net[idx].weight.detach().cpu().flatten().numpy()

        return {
            "train_logits": train_logits.numpy(),
            "ref_logits": ref_logits.numpy(),
            "labels": labels.numpy(),
            "error_matrix": error_matrix.numpy(),
            "error_consistency": error_consistency_kappa(
                train_logits, ref_logits, labels
            ).item(),
            "weights": {
                layer: {
                    "Reference": weights(model_ref, idx),
                    "Trained": weights(model_train, idx),
                }
                for layer, idx in (("net.9", 9), ("net.11", 11))
            },
            "cka_scores": {
                layer_name: float(np.mean(scores))
                for layer_name, scores in cka_values.items()
            },
        }

    analysis_data = _collect_analysis_data()
    return (analysis_data,)


@app.cell
def _(
    MEAN_LINE_DASHES,
    TARGET_SCORE,
    analysis_data,
    class_names,
    colors,
    mo,
    np,
    plotting,
    plt,
    results,
    selected_true_class,
):
    _class_idx = class_names.index(selected_true_class.value)
    _class_mask = analysis_data["labels"] == _class_idx
    _x = np.arange(len(class_names))
    _train_mean_logits = analysis_data["train_logits"][_class_mask].mean(axis=0)
    _ref_mean_logits = analysis_data["ref_logits"][_class_mask].mean(axis=0)
    _error_matrix = analysis_data["error_matrix"]
    _error_share = _error_matrix / _error_matrix.sum()

    # Palette and axis style of the abstract's figure: the trained network
    # in blue, the reference in grey, linear CKA in the measure's blue.
    _train_color = colors.COLOR_BLUE
    _ref_color = colors.COLOR_GREY_MID
    _cka_color = colors.blue_for_measure("cka")
    _line_color = colors.COLOR_GREY_LIGHT
    _violin_width, _half_gap, _half_pad = 0.64, 0.01, 0.03

    with plt.rc_context({"font.size": plotting.FONT_LABEL}):
        # dpi 200 so the small fonts of the figure style stay legible in the
        # notebook (marimo shows the figure at twice the size).
        _fig, _axes = plt.subplots(
            2,
            2,
            figsize=(7.0, 5.8),
            dpi=200,
            constrained_layout=True,
        )
        # Spacing between panels (pads in inches, spaces as a fraction of
        # the panel size).
        _fig.get_layout_engine().set(
            h_pad=0.2,
            w_pad=0.2,
            hspace=0.16,
            wspace=0.1,
        )
        for _ax in _axes.flat:
            plotting.style_axis(_ax)

        # A: mean logits of the test images of the selected true class
        _ax = _axes[0, 0]
        _ax.axhline(0.0, color=_line_color, linewidth=0.8, zorder=1)
        _ax.plot(
            _x,
            _ref_mean_logits,
            color=_ref_color,
            linewidth=2.4,
            marker="o",
            markersize=4.5,
            zorder=2,
        )
        _ax.plot(
            _x,
            _train_mean_logits,
            color=_train_color,
            linewidth=1.2,
            marker="o",
            markersize=2.5,
            zorder=3,
        )
        _ax.set_xticks(_x)
        _ax.set_xticklabels(
            class_names,
            rotation=35,
            ha="right",
            rotation_mode="anchor",
        )
        _ax.set_xlabel(
            f"Output class (test images of class {selected_true_class.value})",
            labelpad=2,
        )
        _ax.set_ylabel("Mean logit", labelpad=2)
        plotting.add_header_legend(
            _ax,
            [(_train_color, "Trained"), (_ref_color, "Reference")],
        )

        # B: error-consistency table (counts of jointly right / wrong trials)
        _ax = _axes[0, 1]
        _ax.imshow(
            _error_share,
            cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list(
                "trained", ["#ffffff", _train_color]
            ),
            vmin=0.0,
            vmax=float(_error_share.max()),
            aspect="auto",
        )
        _ax.axhline(0.5, color="white", linewidth=3)
        _ax.axvline(0.5, color="white", linewidth=3)
        for _row in range(2):
            for _col in range(2):
                _share = _error_share[_row, _col]
                _ax.text(
                    _col,
                    _row,
                    f"{int(_error_matrix[_row, _col]):,}\n{_share:.1%}",
                    ha="center",
                    va="center",
                    color="white"
                    if _share > 0.5 * _error_share.max()
                    else colors.COLOR_GREY_DARK,
                )
        _ax.set_xticks([0, 1])
        _ax.set_xticklabels(["correct", "wrong"])
        _ax.set_yticks([0, 1])
        _ax.set_yticklabels(["correct", "wrong"], rotation=90, va="center")
        _ax.set_xlabel("Trained", labelpad=2)
        _ax.set_ylabel("Reference", labelpad=2)
        for _spine in _ax.spines.values():
            _spine.set_visible(False)
        _ax.tick_params(length=0)
        _ax.set_title(
            "Error consistency, Cohen's $\\kappa$ = "
            f"{analysis_data['error_consistency']:.3f}",
            fontsize=plotting.FONT_LABEL,
            pad=7,
        )

        # C: classifier weights as paired half violins (as Fig. 1B)
        _ax = _axes[1, 0]
        _ax.grid(visible=False)
        _layers = ["net.9", "net.11"]
        _centers = [1.0, 2.0]
        _train_stats = [
            results.violin_statistics(
                analysis_data["weights"][_layer]["Trained"], points=400
            )
            for _layer in _layers
        ]
        _ref_stats = [
            results.violin_statistics(
                analysis_data["weights"][_layer]["Reference"], points=400
            )
            for _layer in _layers
        ]
        _train_parts = _ax.violin(
            _train_stats,
            positions=_centers,
            widths=_violin_width,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        _ref_parts = _ax.violin(
            _ref_stats,
            positions=_centers,
            widths=_violin_width,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for _center, _body in zip(
            _centers, _train_parts["bodies"], strict=True
        ):
            _vertices = np.asarray(_body.get_paths()[0].vertices)  # in place
            _vertices[:, 0] = np.minimum(_vertices[:, 0], _center)
            _body.set_facecolor(_train_color)
            _body.set_edgecolor(_line_color)
            _body.set_alpha(0.92)
            _body.set_linewidth(0.8)
            _body.set_zorder(3)
        for _center, _body in zip(_centers, _ref_parts["bodies"], strict=True):
            _vertices = np.asarray(_body.get_paths()[0].vertices)  # in place
            _vertices[:, 0] = np.maximum(_vertices[:, 0], _center)
            _body.set_facecolor(_ref_color)
            _body.set_edgecolor(_line_color)
            _body.set_alpha(0.78)
            _body.set_linewidth(0.8)
            _body.set_zorder(2)
        for _center, _stats in zip(_centers, _train_stats, strict=True):
            _ax.hlines(
                _stats["mean"],
                _center - _violin_width / 2.0 + _half_pad,
                _center - _half_gap,
                color=_line_color,
                linewidth=0.9,
                linestyles=MEAN_LINE_DASHES,
                zorder=5,
            )
        for _center, _stats in zip(_centers, _ref_stats, strict=True):
            _ax.hlines(
                _stats["mean"],
                _center + _half_gap,
                _center + _violin_width / 2.0 - _half_pad,
                color=_line_color,
                linewidth=0.9,
                linestyles=MEAN_LINE_DASHES,
                zorder=5,
            )
        _ax.set_xlim(_centers[0] - 0.45, _centers[-1] + 0.45)
        _ax.set_xticks(_centers)
        _ax.set_xticklabels(_layers)
        _ax.set_xlabel("Layer", labelpad=2)
        _ax.set_ylabel("Weight value", labelpad=2)
        plotting.add_header_legend(
            _ax,
            [(_train_color, "Trained"), (_ref_color, "Reference")],
        )

        # D: linear CKA to the reference, layer by layer
        _ax = _axes[1, 1]
        _cka_layers = ["early layers", "net.9", "net.10", "net.11"]
        _cka_scores = [1.0] + [
            analysis_data["cka_scores"][_layer] for _layer in _cka_layers[1:]
        ]
        _positions = np.arange(len(_cka_layers))
        _ax.axhline(
            TARGET_SCORE,
            color=_line_color,
            linewidth=0.9,
            linestyle=MEAN_LINE_DASHES,
            zorder=1,
        )
        _ax.text(
            0.02,
            TARGET_SCORE,
            f"target {TARGET_SCORE:.1f}",
            transform=_ax.get_yaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=plotting.FONT_SMALL,
            color=colors.COLOR_GREY_DARK,
        )
        _ax.plot(
            _positions,
            _cka_scores,
            color=_cka_color,
            linewidth=1.6,
            marker="o",
            markersize=4.0,
            zorder=3,
        )
        _ax.set_xlim(-0.4, len(_cka_layers) - 0.6)
        _ax.set_ylim(-0.03, 1.1)
        _ax.set_yticks(np.linspace(0.0, 1.0, 5))
        _ax.set_xticks(_positions)
        _ax.set_xticklabels(
            ["early layers", "net.9", "net.10\n(constrained)", "net.11"]
        )
        _ax.set_xlabel("Layer", labelpad=2)
        _ax.set_ylabel("Linear CKA to the reference", labelpad=2)
        plotting.add_header_legend(_ax, [(_cka_color, "Linear CKA")])

        plotting.add_panel_labels(
            _fig,
            {
                _axes[0, 0]: "A",
                _axes[0, 1]: "B",
                _axes[1, 0]: "C",
                _axes[1, 1]: "D",
            },
        )

    mo.vstack([selected_true_class, _fig])
    return


if __name__ == "__main__":
    app.run()
