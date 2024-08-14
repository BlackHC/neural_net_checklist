# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import neural_net_checklist.torch_diagnostics as torch_diagnostics


# %%
# Define a simple causal transformer model
class CausalTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=4, num_layers=5):
        super(CausalTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(
            16, d_model
        )  # Assuming max sequence length of 16
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward=128, batch_first=True
            ),
            num_layers,
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_encoder(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        x = self.transformer(x, mask, is_causal=True)
        return self.fc_out(x)


# Define a simple tokenizer and vocabulary builder
def simple_tokenizer(text):
    return list(text)


def build_vocab(text):
    tokens = simple_tokenizer(text)
    vocab = {"<pad>": 0, "<unk>": 1}
    for token in tokens:
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


# Define a simple dataset
class TextDataset(Dataset):
    def __init__(self, text, seq_length):
        self.text = text
        self.seq_length = seq_length
        self.vocab = build_vocab(text)
        self.data = [
            self.vocab.get(token, self.vocab["<unk>"])
            for token in simple_tokenizer(text)
        ]

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        inputs = torch.tensor(self.data[idx : idx + self.seq_length])
        targets = torch.tensor(self.data[idx + 1 : idx + self.seq_length + 1])
        return inputs, targets

    def __repr__(self):
        return "TextDataset(" + str(self.data) + ")"


# Sample text
text = """
PRINCE
Rebellious subjects, enemies to peace,
Profaners of this neighbor-stained steel--
Will they not hear?--What ho! You men, you beasts,
That quench the fire of your pernicious rage
With purple fountains issuing from your veins:
On pain of torture, from those bloody hands
Throw your mistempered weapons to the ground,
And hear the sentence of your moved prince.
Three civil brawls bred of an airy word
By thee, old Capulet, and Montague,
Have thrice disturbed the quiet of our streets
And made Verona's ancient citizens
Cast by their grave-beseeming ornaments
To wield old partisans in hands as old,
Cankered with peace, to part your cankered hate.
If ever you disturb our streets again,
Your lives shall pay the forfeit of the peace.
For this time all the rest depart away.
You, Capulet, shall go along with me,
And, Montague, come you this afternoon
To know our farther pleasure in this case,
To old Free-town, our common judgment-place.
Once more, on pain of death, all men depart.
[All but Montague, Lady Montague,
and Benvolio exit.]

MONTAGUE, [to Benvolio]
Who set this ancient quarrel new abroach?
Speak, nephew, were you by when it began?

BENVOLIO
Here were the servants of your adversary,
And yours, close fighting ere I did approach.
I drew to part them. In the instant came
The fiery Tybalt with his sword prepared,
Which, as he breathed defiance to my ears,
He swung about his head and cut the winds,
Who, nothing hurt withal, hissed him in scorn.
While we were interchanging thrusts and blows
Came more and more and fought on part and part,
Till the Prince came, who parted either part.
"""

# Create dataset and dataloader
seq_length = 16
dataset = TextDataset(text, seq_length)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize model
vocab_size = len(dataset.vocab)
model = CausalTransformer(vocab_size)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# %%
# Run diagnostics
torch_diagnostics.ModelInputs(torch_diagnostics.get_supervised_batch(dataloader))
#%%
torch_diagnostics.assert_all_for_causal_llm_cross_entropy_loss(
    lambda: CausalTransformer(vocab_size),
    dataloader,
    embedding_layer_name="embedding",
    vocab_size=vocab_size,
    device="cpu",
)

# # %%
# from tqdm.auto import tqdm

# # Training loop (uncomment to train)
# num_epochs = 100
# for epoch in tqdm(range(num_epochs)):
#     for batch in dataloader:
#         inputs, targets = batch
#         inputs, targets = inputs.to(device), targets.to(device)

#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
#         loss.backward()
#         optimizer.step()

#     if (epoch + 1) % 10 == 0:
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# # %%
# # Generate text (uncomment to generate)
# model.eval()
# with torch.no_grad():
#     input_seq = torch.tensor([dataset.vocab.get(token, dataset.vocab['<unk>']) for token in simple_tokenizer('Throw your')]).unsqueeze(0).to(device)
#     for _ in range(16):
#         output = model(input_seq)
#         next_token = output[:, -1, :].argmax(dim=-1)
#         input_seq = torch.cat([input_seq, next_token.unsqueeze(1)], dim=1)

#     generated_text = ''.join([list(dataset.vocab.keys())[idx] for idx in input_seq.squeeze().tolist()])
#     print(generated_text)

# %%
