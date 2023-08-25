import random
import json
import torch
from model import Net
from nltk_utils import bag_of_words, tokenize

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('Chatbot\Data\intents.json', 'r') as f:
    intents = json.load(f)

FILE_PATH = 'Chatbot\chatmodel.pth'
data = torch.load(FILE_PATH)

input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

model = Net(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = 'Chatgpt'
# print("Let's chat ! (typr 'quit' to exit)")

# while True:
#     sentence = input('You: ')
#     if sentence == 'quit':
#         break
#     sentence = tokenize(sentence)
#     X = bag_of_words(sentence, all_words)
#     X = X.reshape(1, X.shape[0])
#     X = torch.from_numpy(X).to(device)

#     output = model(X)
#     # print(output)
#     _, predicted = torch.max(output, dim=1)
#     # print(predicted)
#     tag = tags[predicted.item()]
#     probs = torch.softmax(output, dim=1)
#     # print(probs)
#     prob = probs[0][predicted.item()]
#     if prob.item() > 0.75:
#         for intent in intents['intents']:
#             if tag == intent['tag']:
#                 print(f"{bot_name}: {random.choice(intent['responses'])}")
#     else:
#         print(f"{bot_name}: I do not understand...")


def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    # print(output)
    _, predicted = torch.max(output, dim=1)
    # print(predicted)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    # print(probs)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                return random.choice(intent['responses'])
    else:
        return 'I do not understand...'
