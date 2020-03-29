import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        #pass
        super(DecoderRNN,self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.word_embed = nn.Embedding(self.vocab_size,self.embed_size)
        
        self.lstm = nn.LSTM(input_size=self.embed_size,hidden_size=self.hidden_size,num_layers=self.num_layers,
                            batch_first=True)
        
        self.hidden_to_vocab = nn.Linear(self.hidden_size,self.vocab_size)
        
        
        
    def forward(self, features, captions):
        #pass
        embeds = self.word_embed(captions[:,:-1])        
        inputs = torch.cat((features.unsqueeze(dim=1),embeds),dim=1)        
        lstm_out,_ = self.lstm(inputs)        
        outputs = self.hidden_to_vocab(lstm_out)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        caption = []
        hidden = (torch.randn(self.num_layers,1,self.hidden_size).to(inputs.device),
                 torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device))
        
        for i in range(max_len):
            lstm_out,hidden = self.lstm(inputs,hidden)
            outputs = self.hidden_to_vocab(lstm_out)
            outputs = outputs.squeeze(1)
            wordid = outputs.argmax(dim=1)
            caption.append(wordid.item())
            
            inputs = self.word_embed(wordid.unsqueeze(0))
            
        return caption