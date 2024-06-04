from synset.base_synset import BaseSynSet
import torch
import copy

class SynSetLoader():

    def __init__(self, synset:BaseSynSet, max_items:int, batch_size:int, copy_synset:bool=True):
        if copy_synset:
            self.synset = copy.deepcopy(synset)
            self.synset.eval()
        else:
            self.synset = synset
        
        self.max_items = max_items
        self.batch_size = batch_size

    def getbatch(self):
        num_items = 0
        unfinished = True
        batch_idx = 0
        while unfinished:
            with torch.no_grad():
                out = self.synset.batch(batch_idx, self.batch_size, tracked=False)
                btch = min(self.synset.num_items, self.batch_size)
                num_items += btch
                if num_items >= self.max_items:
                    num_left = self.max_items-(num_items-btch)
                    out = (ele[:num_left] for ele in out)
                    unfinished = False
                    self.synset.shuffle()
            yield out

    def __iter__(self):
        return self.getbatch()
            
        
