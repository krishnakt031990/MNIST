from models.net import Net
import torch

'''
NOTE: this has evolved through my complete developement process 
to verify if each layer was performing as expected with a random input
of the same dimensions as the one we are expecting with actual images.
'''
def test_train_net_output():
    net = Net()
    x = torch.randn(1, 1, 28, 28)
    y = net(x)
    assert y.size() == (1, 10)


test_train_net_output()
