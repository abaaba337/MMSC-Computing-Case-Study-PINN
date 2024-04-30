
import torch
import torch.nn as nn

class rff(nn.Module):
    
    def __init__(self, input_num, rff_num, u=0, std=1, rff_B=None):  
        super().__init__()
        if rff_B == None:
            self.B = torch.randn(rff_num,input_num) * std + u
        else:
            self.B = rff_B
    
    def forward(self, x): 
        Bv = 2 * torch.pi * torch.matmul(self.B, x.T).T
        rff_layer_output = torch.concat((torch.cos(Bv) , torch.sin(Bv)), dim=1)
        return rff_layer_output