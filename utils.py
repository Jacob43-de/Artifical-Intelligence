import torch
import collections
import jieba
from torch.utils import data

class Vocab:
    
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None) -> None:
        if tokens == None:
            tokens =[]
        if reserved_tokens == None:
            reserved_tokens = []
        
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        
        self.idx_to_token =  reserved_tokens + ['<unk>']
        # print(self.idx_to_token)
        self.token_to_idx = {token:idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
    
    def __len__(self):
        return len(self.idx_to_token)
    
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
    
    @property        
    def unk(self):
        return 0
    
    @property
    def token_freqs(self):
        return self._token_freqs
    
def count_corpus(tokens): 
    """Count token frequencies."""
    # print(tokens)
    # Here `tokens` is a 1D list or 2D list
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of token lists into a list of tokens
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

def truncate_pad(line, num_steps, padding_token):
    """ truncate or pad a sequence """
    
    if len(line) > num_steps:
        return line[:num_steps]
    else:
        return line + [padding_token] * (num_steps-len(line))

def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator."""
    
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def splitTrain_test(dataset, training_rate=0.9):
    """
        return: training set, test set
    """
    num_samples = len(dataset)
    num_training = int(num_samples * training_rate)
    # num_testing = num_samples - num_training
    
    return dataset[:num_training], dataset[num_training:]

def load_SA_data(pos_dir, neg_dir, batch_size, max_len):
    
    # 1 denotes the positive, 0 denotes the negative
    pos_data = [(line.strip(), 1) for line in open(pos_dir, 'rt', encoding='UTF-8')]
    neg_data = [(line.strip(), 0) for line in open(neg_dir, 'rt', encoding='UTF-8')]
    
    dataset_labels = pos_data + neg_data
    dataset = [sent[0] for sent in dataset_labels]
    labels = [sent[1] for sent in dataset_labels]
    # print(type(dataset))
    dataset = tokenize_sentences(dataset)
    vocab = Vocab(dataset, min_freq=1, reserved_tokens=['<pad>', '<sos>', '<eos>'])
    
    train_data, test_data = splitTrain_test(dataset)
    train_label, test_label = splitTrain_test(labels)
    
    trainData_idxs = torch.tensor([truncate_pad(vocab[sent], max_len, vocab['<pad>']) for sent in train_data])
    testData_idxs = torch.tensor([truncate_pad(vocab[sent], max_len, vocab['<pad>']) for sent in test_data])
    
    train_iter = load_array((trainData_idxs, torch.tensor(train_label)), batch_size)
    test_iter = load_array((testData_idxs, torch.tensor(test_label)), batch_size, is_train=False)
    
    return train_iter, test_iter, vocab
    

def tokenize_sentences(input_data):
    
    tokenized_sents = []
    for sent in input_data:
        tokenized_sents.append(tokenize(sent))
    
    return tokenized_sents

def tokenize(sentence):
    
    sentence = sentence.strip()
    words = jieba.lcut(sentence, cut_all=False)
    return words

if __name__ == '__main__':
    
    data_dir = 'E:\projects\ZX-projs\sentiment-analysis\data\shopping reviews\\'
    pos_file = 'pos.csv'
    neg_file = 'neg.csv'
    
    train_iter, test_iter, vocab = load_SA_data(data_dir+pos_file, data_dir+neg_file, 16, 100)
    # print(vocab)
    # print(train_iter)
    for i, (feature, label) in enumerate(train_iter):
        
        # if i == 10:
        #     break
        
        print(i, feature.shape, label.shape)
        break