from torch.utils.data._utils.collate import default_collate


def custom_collate_fn(batch):
    inputs = [item[0] for item in batch]
    annotations = [tuple(item[1]) for item in batch]

    # Stack input tensors
    inputs = default_collate(inputs)

    # Keep annotations as a list or convert to appropriate format
    return inputs, annotations
