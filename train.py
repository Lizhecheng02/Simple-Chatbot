import numpy as np
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_utils import bag_of_words, tokenize, stem
from model import Net
from tqdm import tqdm

with open('Chatbot\Data\intents.json', 'r') as f:
    intents = json.load(f)
    # print(intents)

all_words = []
tags = []
Xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        Xy.append((w, tag))

ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# print(all_words)
# print(tags)

# print(len(Xy), 'patterns')
# print(len(tags), 'tags:', tags)
# print(len(all_words), 'unique stemmed words:', all_words)


X_train = []
y_train = []
for (pattern_sentence, tag) in Xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

# print(X_train)
# print(y_train)

X_train = np.array(X_train)
y_train = np.array(y_train)

num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 16
output_size = len(tags)
# print(input_size, output_size)


class ChatDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.n_samples = len(X_train)
        self.X_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


dataset = ChatDataset()
train_loader = DataLoader(
    dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net(input_size=input_size, hidden_size=hidden_size,
            num_classes=output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in tqdm(range(num_epochs)):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1} / {num_epochs}], Loss: {loss.item(): .4f}')

data = {
    'model_state': model.state_dict(),
    'input_size': input_size,
    'hidden_size': hidden_size,
    'output_size': output_size,
    'all_words': all_words,
    'tags': tags
}

FILE_PATH = 'Chatbot\chatmodel.pth'
torch.save(data, FILE_PATH)

print('Save successfully!!!')
