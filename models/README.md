# Model checkpoints

| File                              | Contents                                                                                                                                                                                                                                                                              |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lenet_fashionmnist_reference.pt` | The fixed reference LeNet-5 of the demo (`notebooks/demo/lenet_fashionmnist.py`), trained with the recipe in that notebook (SGD, 80 epochs with early stopping at epoch 44, best validation accuracy 0.924 at epoch 34). Stored as `{"epoch", "val_loss", "val_acc", "model_state"}`. |

The ResNet-50 and AlexNet checkpoints behind Figure 1 are too large for the
repository and are archived separately; see `results/README.md`.
