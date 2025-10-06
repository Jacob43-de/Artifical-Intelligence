import torch
from torch import nn
from d2l import torch as d2l
from sa_model import BiRNN
from utils import load_SA_data, tokenize

batch_size = 64
max_len = 50
data_dir = './data/'
pos_file = 'pos.csv'
neg_file = 'neg.csv'
lr, num_epoch = 0.01, 5


embed_size, num_hiddens, num_layers, devices = 300, 256, 2, d2l.try_all_gpus()

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Embedding:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.LSTM:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])

def train_batch(net, X, y, loss, trainer, devices):
    """Train for a minibatch with mutiple GPUs (defined in Chapter 13)."""
    if isinstance(X, list):
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum

def train_epochs(net, train_iter, test_iter, loss, trainer, num_epochs, devices=d2l.try_all_gpus()):
        
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
    legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples, no. of predictions
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch(net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                print('batch: {0}, loss: {1}, accuracy: {2}'.format(i + 1, metric[0] / metric[2], metric[1] / metric[3]))
                # animator.add(epoch + (i + 1) / num_batches, (metric[0] / metric[2], metric[1] / metric[3], None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        print('epoch: {0}, test accuracy: {1}'.format(epoch + 1, test_acc))
        # animator.add(epoch + 1, (None, None, test_acc))
            
    print(f'loss {metric[0] / metric[2]:.3f}, train acc ' f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on ' f'{str(devices)}')

def predict_sentiment(net, sequence):
    
    """Predict the sentiment of a text sequence."""
    
    # sequence = torch.tensor(vocab[sequence.split()], device=d2l.try_gpu())
    label = torch.argmax(net(sequence.reshape(1, -1)), dim=1)
    
    return 'positive' if label == 1 else 'negative'

if __name__ == '__main__':
    
    train_iter, test_iter, vocab = load_SA_data(data_dir+pos_file, data_dir+neg_file, batch_size, max_len)
    net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction="none")
    net.apply(init_weights)
    
    train_epochs(net, train_iter, test_iter, loss, trainer, num_epoch, devices)
    
    for i, (feat, label) in enumerate(test_iter):
        for i in range(len(feat)):
            one_sample = feat[i,:]
            # print(one_sample.tolist())
            print('sent: ', ''.join(vocab.to_tokens(one_sample.tolist())), 'label: ', 'pos' if label[i].tolist() == 1 else 'neg', 'predict: ', predict_sentiment(net, one_sample))
        break
