import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from src.nn_rff import rff
from src.unlabeled_dataset import UnlabeledDataset
from torch.utils.data import DataLoader


class PDE_PINN_HARDBC_ELLIPTIC(nn.Module): 
    
    ## Initialize 
    def __init__(self, f, rec_bound, BC, 
                 n_hidden, n_layers,
                 set_rff=False, rff_num=25, u=0, std=1, rff_B=None):
        
        super().__init__()
        self.model_type = 'PDE_PINN_HARDBC_ELLIPTIC'
        self.f = f
        self.rec_bound , self.BC = rec_bound , BC

        self.train_loss , self.validate_loss , self.L2_loss = [] , [] , []
        self.n_hidden  , self.n_layers = n_hidden , n_layers
        
        self.activation = nn.Tanh
        self.rff_para = (set_rff, rff_num, u, std)
        self.rff_B = rff_B
        
        if set_rff :
            self.rff = rff(2, rff_num, u, std, rff_B)
            self.input  = nn.Sequential( self.rff,
                                         nn.Linear(2*rff_num, self.n_hidden), 
                                         self.activation() )
            self.rff_B = self.rff.B
        else :
            self.input  = nn.Sequential( nn.Linear(2, self.n_hidden), self.activation() )
            
        self.hiddens = nn.Sequential( *[ nn.Sequential( nn.Linear(self.n_hidden, self.n_hidden), self.activation() ) 
                                        for _ in range(self.n_layers-3)] )
        self.output  =  nn.Linear(self.n_hidden, 1)
        self.fwd     =  nn.Sequential(self.input, self.hiddens, self.output)
        
        
    
    
    ## Construct dataloader
    def sample_data_points(self, n_interior=1000):
        x_interior_tensor = torch.rand(n_interior, 2, dtype=torch.float32)    
        x_interior_tensor[:,0] = ( self.rec_bound[1] - self.rec_bound[0] ) * x_interior_tensor[:,0] + self.rec_bound[0]
        x_interior_tensor[:,1] = ( self.rec_bound[3] - self.rec_bound[2] ) * x_interior_tensor[:,1] + self.rec_bound[2]     
        x_interior_tensor.requires_grad = True
        return x_interior_tensor
    
    def construct_train_dataloader(self, train_batch_size=32, n_interior=1000):
        x_tensor = self.sample_data_points(n_interior)
        dataset = UnlabeledDataset(x_tensor)
        loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)
        return loader
    
    def sample_one_batch(self, batch_size=32, random_seed=-1):
        if random_seed > 0:
            torch.manual_seed(random_seed)
        x_tensor =  self.sample_data_points(batch_size)
        return x_tensor

    def generate_grid(self, n_each_axis=100):
        x = np.linspace(self.rec_bound[0], self.rec_bound[1], n_each_axis)
        y = np.linspace(self.rec_bound[2], self.rec_bound[3], n_each_axis)
        X, Y = np.meshgrid(x, y)
        XY = np.column_stack((X.flatten(),Y.flatten()))
        XY_tensor = torch.tensor(XY, dtype=torch.float32)
        return XY_tensor
        


    
    ## Evaluate Loss
    def ResidualLoss(self, x, u_hat):
        Du_hat  = torch.autograd.grad(u_hat, x,
                                      grad_outputs=torch.ones_like(u_hat), 
                                      create_graph=True)[0]      # 1st derivative, [ batch_size , 2 ]

        Dxxu_hat = torch.autograd.grad(Du_hat[:,0], x,
                                       grad_outputs=torch.ones_like(Du_hat[:,0]),
                                       create_graph=True)[0][:,0].unsqueeze(dim=1)  # 2nd derivative, [ batch_size , 1 ]
                
        Dyyu_hat = torch.autograd.grad(Du_hat[:,1], x,
                                       grad_outputs=torch.ones_like(Du_hat[:,1]),
                                       create_graph=True)[0][:,1].unsqueeze(dim=1)  # 2nd derivative, [ batch_size , 1 ]
        f_hat = self.f(x)
        Lp = ((Dxxu_hat + Dyyu_hat - f_hat)**2).mean()
        return Lp
    
    def L2_error(self, true_sol, n_each_axis=100):
        self.eval()
        XY = self.generate_grid(n_each_axis)
        X = XY[:,0].reshape(n_each_axis,n_each_axis)[0,:]
        Y = XY[:,1].reshape(n_each_axis,n_each_axis)[:,0]
        U = self.forward(XY)
        U_true = true_sol(XY)
        SqError = (U-U_true)**2
        SqError = SqError.reshape(n_each_axis,n_each_axis)
        
        slice = []
        for i in range(len(X)) :
            Y_slice =  torch.trapz(SqError[:,i], Y)
            slice.append(Y_slice.item())
        res = torch.trapz(torch.tensor(slice, dtype=torch.float32), X)
        return (res.item())**0.5
    
    
    
    
    ## Forward pass function
    def forward(self, x): 
        x_ , y_ = x[:,0].unsqueeze(1) , x[:,1].unsqueeze(1)
        a , b , c , d = self.rec_bound
        
        a_ = a * torch.ones_like(x_, requires_grad=True)
        b_ = b * torch.ones_like(x_, requires_grad=True)
        c_ = c * torch.ones_like(x_, requires_grad=True)
        d_ = d * torch.ones_like(x_, requires_grad=True)
        
        u_ac = self.fwd(torch.cat((a_, c_), dim=1))
        u_bd = self.fwd(torch.cat((b_, d_), dim=1))
        u_bc = self.fwd(torch.cat((b_, c_), dim=1))
        u_ad = self.fwd(torch.cat((a_, d_), dim=1))
        
        u_ay = self.fwd(torch.cat((a_, y_), dim=1))
        u_by = self.fwd(torch.cat((b_, y_), dim=1))
        u_xc = self.fwd(torch.cat((x_, c_), dim=1))
        u_xd = self.fwd(torch.cat((x_, d_), dim=1))
        
        u_ay_true = self.BC[0](y_) ; u_by_true = self.BC[1](y_)
        u_xc_true = self.BC[2](x_) ; u_xd_true = self.BC[3](x_)
        
        u_ac_true = self.BC[0](c_) ; u_bc_true = self.BC[1](c_)
        u_ad_true = self.BC[0](d_) ; u_bd_true = self.BC[1](d_)
        
        u = self.fwd(x) + ( b - x_ ) / ( b - a ) * ( u_ay_true - u_ay ) + ( x_ - a ) / ( b - a ) * ( u_by_true - u_by ) \
              + ( d - y_ ) / ( d - c ) * ( u_xc_true - u_xc ) + ( y_ - c ) / ( d - c ) * ( u_xd_true - u_xd ) \
              - ( b - x_ ) / ( b - a ) * ( d - y_ ) / ( d - c ) * ( u_ac_true - u_ac) \
              - ( x_ - a ) / ( b - a ) * ( d - y_ ) / ( d - c ) * ( u_bc_true - u_bc) \
              - ( b - x_ ) / ( b - a ) * ( y_ - c ) / ( d - c ) * ( u_ad_true - u_ad) \
              - ( x_ - a ) / ( b - a ) * ( y_ - c ) / ( d - c ) * ( u_bd_true - u_bd) 
        return u 
    
    

    ## Validation function
    def Validate(self, sample_num, random_seed=-1):
        self.eval()
        x = self.sample_one_batch(sample_num, random_seed)
        u_hat = self.forward(x) 
        loss = self.ResidualLoss(x, u_hat)           
        return loss.item()
    
    ## Test function
    def Test(self, sample_num, random_seed=-1):
        test_loss = self.Validate(sample_num, random_seed)
        print( 'Test set: Avg. Test Sample Loss: {:.4f}'.format(test_loss) )
        return test_loss

    ## Training function
    def Train(self, train_num, train_batch_size, learning_rate,  
              lr_step_size=100, min_lr =1e-10, lr_gamma=0.5,
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
                u_hat = self.forward(x)   
                loss = self.ResidualLoss(x, u_hat)         # evaluate loss
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
                scheduler.step()     