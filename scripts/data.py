from torch.utils.data import Dataset
from prep_data import format_input


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            
            # encode fully formatted data point
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, ind):
        return self.encoded_texts[ind]

    def __len__(self):
        return len(self.data)