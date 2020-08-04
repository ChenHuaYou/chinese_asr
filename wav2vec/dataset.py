from torch.utils.data import Dataset,DataLoader
import torch
from pathlib import Path
from pydub import AudioSegment
import soundfile as sf



class LibriSpeechDataset(Dataset):

    def __init__(self,data_dir):
        p = Path(data_dir)
        self.examples = []
        for name in p.glob('**/*.txt'):
            for line in open(name,'r'):
                lineno,labels = line.split(' ',1)
                d1,d2,d3 = lineno.split('-')
                path = p/d1/d2/(lineno+'.flac')
                example = {'audio':str(path),'labels':labels}
                self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self,idx):
        example = self.examples[idx]
        left = example['audio']
        right = example['labels']

        return left, right

def LibriSpeechDataLoader(data_dir):
    dataset = LibriSpeechDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=None)
    return dataloader


if __name__ == '__main__':

    dataloader = LibriSpeechDataLoader('./train/LibriSpeech/train-clean-100')
    for i_batch, sample_batched in enumerate(dataloader):
        path = sample_batched[0][0]
        data, samplerate = sf.read(path)                                            
        print(data)
