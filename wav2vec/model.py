import torch
from fairseq.models.wav2vec import Wav2VecModel
import torch.nn as nn
from ctc import decode

cuda = torch.device('cuda:0')

class wav2vec_ctc(nn.Module):

    def __init__(self,vocab_size):
        super(wav2vec_ctc, self).__init__()
        cp = torch.load('wav2vec_large.pt')
        self.model = Wav2VecModel.build_model(cp['args'], task=None)
        self.linear = nn.Linear(512,vocab_size)
        self.model.load_state_dict(cp['model'])

    def forward(self,inputs,targets=None):
        ins = []
        outs = []
        max_length = max(len(i) for i in inputs)
        inputs = [i+[0]*(max_length-len(i)) for i in inputs]
        inputs = torch.tensor(inputs).to(cuda)
        x = self.model.feature_extractor(inputs)
        x = x.permute(0,2,1) # (N,C,L)->(N,L,C)
        x = self.linear(x)
        x = x.permute(1,0,2) # (N,L,C) -> (L,N,C)
        x = x.log_softmax(2)
        L = x.shape[0]
        N = x.shape[1]
        input_lengths = tuple([L]*N)

        if targets is not None:
            target_lengths = tuple([len(i) for i in targets])
            max_length = max(target_lengths)
            targets = [i+[0]*(max_length-len(i)) for i in targets]
            targets = torch.tensor(targets).to(cuda)

            ctc_loss = nn.CTCLoss()
            loss = ctc_loss(x, targets, input_lengths, target_lengths)
            return loss

        return x



