import json

def format_input(entry):
    """
    format each point into the Alpaca format, made up of 3 components:

    ### Instruction:    
    ### Input:
    ### Response
    """
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction: \n{entry['instruction']}"
    )
    # points can have no input, e.g.
    #  {'instruction': "What is an antonym of 'complicated'?", 'input': '', 'output': "An antonym of 'complicated' is 'simple'."}
    input_text = f"\n\n###Input: \n{entry['input'] if entry['input'] else ''}"
    
    return instruction_text + input_text


def load_and_split_data(data_paths, train_split=0.85, test_split=0.1):
    """Load and split data into train, validation and test sets."""
    data = []
    for path in data_paths:
        with open(path, 'r') as f:
            data.extend(json.load(f))
    
    N = len(data)
    print(f'Total data: {N}')

    train_portion = int(N * train_split)
    test_portion = int(N * test_split)
    val_portion = N - train_portion - test_portion

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    print(f"Training set length: {len(train_data)}")
    print(f"Validation set length: {len(val_data)}")
    print(f"Test set length: {len(test_data)}")

    return train_data, val_data, test_data