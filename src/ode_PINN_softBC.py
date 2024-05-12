import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from src.nn_rff import rff
from src.unlabeled_dataset import UnlabeledDataset
from torch.utils.data import DataLoader

## Define the network model
class ODE_PINN_SOFTBC(nn.Module): 
    
    ## Initialize 
    def __init__(self, f, lb, ub, BC, lambdas,
                 n_hidden, n_layers, 
                 set_rff=False, rff_num=25, u=0, std=1, rff_B=None):  # n, tol=0.001, max_epoch=3000, display=True
        
        # f        :  y'' = f(x,y,y') tensor function acted on tensors x and y of size [ batch_size , 1 ]
        
        # lb , ub  :  float, lower and upper bound
        # BC       :  tuple, (type, yl, yu) 
                       # type 1 : y(lb) = yl ,  y(ub) = yu
                       # type 2 : y(lb) = yl ,  y'(ub) = yu
                       # type 3 : y(lb) + y'(lb) = yl ,  y'(ub) = yu
                       
        # lambdas  :  tuple, (type, lambda_p, lambda_b1, lambda_b2) 
        # n_hidden :  width of the hidden layer
        # n_layers :  number of the layers (>=3), n_layers-2 hidden layers
        
        super().__init__()
        
        self.model_type = 'ODE_PINN_SOFTBC'
        self.f = f
        self.lb , self.ub , self.BC = lb , ub , BC
        self.xl = torch.tensor(lb, dtype=torch.float32, requires_grad=True).view(-1,1)
        self.xu = torch.tensor(ub, dtype=torch.float32, requires_grad=True).view(-1,1)

        self.train_loss , self.validate_loss , self.L2_loss = [] , [] , []
        self.lambdas = lambdas
        self.n_hidden  , self.n_layers = n_hidden , n_layers
        
        self.activation = nn.Tanh
        self.rff_para = (set_rff, rff_num, u, std)
        self.rff_B = rff_B
        
        if set_rff :
            self.rff = rff(1, rff_num, u, std, rff_B)
            self.input  = nn.Sequential( self.rff,
                                         nn.Linear(2*rff_num, self.n_hidden), 
                                         self.activation() )
        else :
            self.input  = nn.Sequential( nn.Linear(1, self.n_hidden), self.activation() )
            
        self.hiddens = nn.Sequential( *[ nn.Sequential( nn.Linear(self.n_hidden, self.n_hidden), self.activation() ) 
                                        for _ in range(self.n_layers-3)] )
        self.output  =  nn.Linear(self.n_hidden, 1)
        self.fwd     =  nn.Sequential(self.input, self.hiddens, self.output)
        
        
        
    ## Construct dataloader
    def sample_one_batch(self, batch_size=32, random_seed=-1):
           
        if random_seed > 0:
            torch.manual_seed(random_seed)
        x_tensor = torch.rand(batch_size, 1, dtype=torch.float32)    
        x_tensor = ( self.ub - self.lb ) * x_tensor + self.lb
        x_tensor.requires_grad = True
        
        return x_tensor
    
    
    def construct_train_dataloader(self, train_batch_size, train_sample_num =1000):
           
        x_tensor = torch.rand(train_sample_num, 1, dtype=torch.float32)    
        x_tensor = ( self.ub - self.lb ) * x_tensor + self.lb
        x_tensor.requires_grad = True

        # Construct datasets and dataloader
        dataset = UnlabeledDataset(x_tensor)
        loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)

        return loader
    

    
    ## Forward pass function
    def forward(self, x): 
        return self.fwd(x)
    
    
    ## Evaluate Loss
    def ResidualLoss(self, x, y_hat):
        D1y_hat  = torch.autograd.grad(y_hat, x,
                                      grad_outputs=torch.ones_like(y_hat), 
                                      create_graph=True)[0]     

        D2y_hat = torch.autograd.grad(D1y_hat, x,
                                      grad_outputs=torch.ones_like(D1y_hat),
                                      create_graph=True)[0]
        
        f_hat = self.f(x, y_hat, D1y_hat)
        Lp = ((D2y_hat - f_hat)**2).sum()/len(x)
        

                       # type 2 : y(lb) = yl ,  y'(ub) = yu
                       # type 3 : y(lb) + y'(lb) = yl ,  y'(ub) = yu
                       
                       
        if  self.BC[0] == 1 :
            Lb1 = (self.lambdas[1] * ( self.forward(self.xl)-self.BC[1] )**2).squeeze()
            Lb2 = (self.lambdas[2] * ( self.forward(self.xu)-self.BC[2] )**2).squeeze()
                  
        elif  self.BC[0] == 2 :
            y_hat_l = self.forward(self.xl)
            y_hat_u = self.forward(self.xu)
            
            D1y_hat_u = torch.autograd.grad(y_hat_u, self.xu, torch.ones_like(y_hat_u), create_graph=True)[0]
            
            Lb1 = (self.lambdas[1] * ( y_hat_l - self.BC[1] )**2).squeeze()
            Lb2 = (self.lambdas[2] * ( D1y_hat_u - self.BC[2] )**2).squeeze()
            
        else:
            y_hat_l = self.forward(self.xl)
            y_hat_u = self.forward(self.xu)
            
            D1y_hat_l = torch.autograd.grad(y_hat_l, self.xl, torch.ones_like(y_hat_l), create_graph=True)[0]
            D1y_hat_u = torch.autograd.grad(y_hat_u, self.xu, torch.ones_like(y_hat_u), create_graph=True)[0]
            
            Lb1 = (self.lambdas[1] * ( y_hat_l +  D1y_hat_l - self.BC[1] )**2).squeeze()
            Lb2 = (self.lambdas[2] * ( D1y_hat_u - self.BC[2] )**2).squeeze()
        
        return Lp + Lb1 + Lb2
    
    
    def L2_error(self, true_sol):
        self.eval()
        x = torch.linspace(self.lb,self.ub,2000)
        y_hat  = self.forward(x.view(-1,1))
        y_true = true_sol(x.view(-1,1))
        z = ((y_true - y_hat)**2).squeeze()
        res = torch.trapz(z, x)
        return (res.item())**0.5
    
    
    
    
    ## Validation function
    def Validate(self, sample_num, random_seed=-1):
        self.eval()
        x = self.sample_one_batch(sample_num, random_seed)
        y_hat = self.forward(x) 
        loss = self.ResidualLoss(x, y_hat)
        return loss.item()
    
    
    
    ## Training function
    def Train(self, train_num, train_batch_size, learning_rate, 
              lr_step_size=100, min_lr =5e-4, lr_gamma=0.5,
              abs_tolerance=1e-4, max_epoch=3000, compute_L2_loss=False, true_sol=None, display=True): 
        
        
        train_dataloader = self.construct_train_dataloader(train_batch_size, train_num)
        n_train_batches = len(train_dataloader)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)             # optimizer, adam optimizer
        scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)   # learning updater
        
        for epoch in range(max_epoch): # training starts
            
            if display :
                print('------------------------------------------------------- ')
                print('-------------------- Epoch [{}/{}] -------------------- '.format(epoch + 1, max_epoch))
            
            self.train() ; train_loss = 0

            for i, x in enumerate(train_dataloader):
                
                # forward calculation
                # x = self.sample_one_batch(batch_size=train_batch_size)
                y_hat = self.forward(x)   
                loss = self.ResidualLoss(x, y_hat)         # evaluate loss
                optimizer.zero_grad()                      # clear gradients
                loss.backward()                            # back propgation
                optimizer.step()                           # update parameters
                train_loss += loss.item()                  # compute the total loss for all batches in train set
                
                # Display the training progress
                if display :
                    if (i==0) or ((i+1) % round(n_train_batches/5) == 0):
                        print( 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                            epoch + 1, max_epoch, i + 1, n_train_batches, loss.item()) ) # train sample loss
            
            # Validate the model
            validate_num = round(train_num/3)
            self.validate_loss.append(self.Validate(validate_num))    # record average validate sample loss for each epoch    
            self.train_loss.append(train_loss/n_train_batches)        # record average train sample loss for each epoch    
            
            if compute_L2_loss:
                self.L2_loss.append(self.L2_error(true_sol))
                
            if display :
                if compute_L2_loss:
                    print( 'Epoch [{}/{}], Avg. Train Sample Loss: {:.4f}, Avg. Validate Sample Loss: {:.4f}, \
                            L2 Loss: {:.4f}'.format(
                            epoch + 1, max_epoch, self.train_loss[-1], self.validate_loss[-1], self.L2_loss[-1]) )
                else: 
                    print( 'Epoch [{}/{}], Avg. Train Sample Loss: {:.4f}, Avg. Validate Sample Loss: {:.4f}'.format(
                            epoch + 1, max_epoch, self.train_loss[-1], self.validate_loss[-1]) )
            
            if self.validate_loss[-1] < abs_tolerance :
                break
            elif len(self.validate_loss) > 26 :
                temp_validate_loss = np.array(self.validate_loss)[-1:-27:-1]
                temp_validate_loss_diff = np.diff(temp_validate_loss)
                temp_rel_validate_loss = np.abs(temp_validate_loss_diff/temp_validate_loss[:25])
                if ( temp_rel_validate_loss < 0.0001 ).sum() == 25 :
                    break      
            elif optimizer.param_groups[0]['lr'] > min_lr :
                scheduler.step()                                     # update learning rate   
                    
              
                    
    ## Test function
    def Test(self, sample_num, random_seed=-1):
        self.eval() ; 
        x = self.sample_one_batch(sample_num, random_seed)
        y_hat = self.forward(x) 
        loss = self.ResidualLoss(x, y_hat)       
        test_loss = loss.item()    
        print( 'Test set: Avg. Test Sample Loss: {:.4f}'.format(test_loss) )
        return test_loss
