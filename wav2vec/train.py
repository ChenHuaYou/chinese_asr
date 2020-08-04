from pathlib import Path
from tqdm import tqdm
import torch
from transformers import BertTokenizer
from dataset import LibriSpeechDataLoader
from model import wav2vec_ctc
import soundfile as sf

cuda = torch.device('cuda:0')

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size
    print('vocab_size is %s'%vocab_size)


    dataloader = LibriSpeechDataLoader('./train/LibriSpeech/train-clean-100')
    pbar = tqdm(dataloader)

    model = wav2vec_ctc(vocab_size)
    model.to(cuda)
    model.train()
    check_point_dir = Path('./model/')
    model_file = check_point_dir/'model.pth'
    if model_file.is_file():
        print('model file exists, load it')
        model.load_state_dict(torch.load(model_file))

    lr = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #inputs = [torch.randn(5000).tolist(), torch.randn(10000).tolist()]
    #outputs = ['i love china !','do you hate america ?']
    #targets = [tokenizer.encode(s, add_special_tokens=False) for s in outputs]
    for epoch in range(5):
        epoch_loss = 0
        for i_batch, sample_batched in enumerate(pbar):
            paths = sample_batched[0]
            labels = sample_batched[1]
            inputs = []
            for path in paths: 
                data, samplerate = sf.read(path)                                            
                data = data.tolist()
                inputs.append(data)
            targets = [tokenizer.encode(s, add_special_tokens=False) for s in labels]
            optimizer.zero_grad()
            loss = model(inputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            avg_loss = epoch_loss / (i_batch+1)

            pbar.set_description("train: epoch %s / loss %s"%(epoch, avg_loss))

