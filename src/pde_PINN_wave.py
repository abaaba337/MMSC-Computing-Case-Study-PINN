# 2space wave 
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from src.nn_rff import rff
from src.unlabeled_dataset import UnlabeledDataset
from torch.utils.data import DataLoader


class PDE_PINN_WAVE(nn.Module): 
    
    ## Initialize 
    def __init__(self, paras, BC, lambdas,
                 n_hidden, n_layers,
                 set_rff=False, rff_num=25, u=0, std=1, rff_B=None):
        
        super().__init__()
        self.model_type = 'PDE_PINN_WAVE'
        self.g = BC[0] ; self.phi = BC[1]
        self.c , self.a , self.b , self.T = paras
        self.lambdas = lambdas

        self.train_loss , self.validate_loss , self.L2_loss = [] , [] , []
        self.n_hidden  , self.n_layers = n_hidden , n_layers
        
        self.activation = nn.Tanh
        self.rff_para = (set_rff, rff_num, u, std)
        self.rff_B = rff_B
        
        if set_rff :
            self.rff = rff(3, rff_num, u, std, rff_B)
            self.input  = nn.Sequential( self.rff,
                                         nn.Linear(2*rff_num, self.n_hidden), 
                                         self.activation() )
            self.rff_B = self.rff.B
        else :
            self.input  = nn.Sequential( nn.Linear(3, self.n_hidden), self.activation() )
            
        self.hiddens = nn.Sequential( *[ nn.Sequential( nn.Linear(self.n_hidden, self.n_hidden), self.activation() ) 
                                        for _ in range(self.n_layers-3)] )
        self.output  =  nn.Linear(self.n_hidden, 1)
        self.fwd     =  nn.Sequential(self.input, self.hiddens, self.output)
        
    
    def generate_XYgrid(self, n_each_axis=100):
        x = np.linspace(0, self.a, n_each_axis)
        y = np.linspace(0, self.b, n_each_axis)
        X,Y = np.meshgrid(x,y)
        XY = np.column_stack((X.flatten(), Y.flatten()))
        XY_tensor = torch.tensor(XY, dtype=torch.float32)
        return XY_tensor
    
    def trapz2D(self, f, n_each_axis=100):
        XY = self.generate_XYgrid(n_each_axis)
        X = XY[:,0].reshape(n_each_axis,n_each_axis)[0,:]
        Y = XY[:,1].reshape(n_each_axis,n_each_axis)[:,0]
        F = f(XY).reshape(n_each_axis,n_each_axis)
        slice = []
        for i in range(len(X)) :
            Y_slice =  torch.trapz(F[:,i], Y)
            slice.append(Y_slice.item())
        res = torch.trapz(torch.tensor(slice, dtype=torch.float32), X)
        return res.item()
    
    
    def true_sol(self, xyt, k):
        x = xyt[:,0]
        y = xyt[:,1]
        t = xyt[:,2]
        res = 0

        for n in range(1,k+1):
            for m in range(1,k+1):
                u_m = m*np.pi/self.a
                v_n = n*np.pi/self.b
                lambda_mn = self.c * np.sqrt(u_m**2 + v_n**2)
                Amn = (4/(self.a*self.b)) * self.trapz2D(lambda xy: self.g(xy)*torch.sin(u_m*xy[:,0])*torch.sin(v_n*xy[:,1])
                                                         , n_each_axis=100)
                Bmn = (4/(self.a*self.b*lambda_mn)) * self.trapz2D(lambda xy: self.phi(xy)*torch.sin(u_m*xy[:,0])*torch.sin(v_n*xy[:,1])
                                                                   , n_each_axis=100)
                res += torch.sin(u_m*x)*torch.sin(v_n*y)*( Amn*torch.cos(lambda_mn*t) + Bmn*torch.sin(lambda_mn*t) )
                
        return res.view(-1,1)
            
    
    ## Construct dataloader
    def sample_data_points(self, n_interior=1000, n_boundary_each=200):
        x_interior_tensor = torch.rand(n_interior, 3, dtype=torch.float32)    
        x_interior_tensor[:,0] = self.a  * x_interior_tensor[:,0]
        x_interior_tensor[:,1] = self.b  * x_interior_tensor[:,1] 
        x_interior_tensor[:,2] = self.T  * x_interior_tensor[:,2] 
        x_interior_tensor.requires_grad = True
        
        x0 = torch.rand(n_boundary_each, 3, dtype=torch.float32)    
        x0[:,0] = 0
        x0[:,1] = self.b  * x0[:,1]
        x0[:,2] = self.T  * x0[:,2]
        x0.requires_grad = True
        
        xa = torch.rand(n_boundary_each, 3, dtype=torch.float32)    
        xa[:,0] = self.a
        xa[:,1] = self.b  * xa[:,1]
        xa[:,2] = self.T  * xa[:,2]
        xa.requires_grad = True
        
        y0 = torch.rand(n_boundary_each, 3, dtype=torch.float32)    
        y0[:,0] = self.a  * y0[:,0]
        y0[:,1] = 0
        y0[:,2] = self.T  * y0[:,2]
        y0.requires_grad = True
        
        yb = torch.rand(n_boundary_each, 3, dtype=torch.float32)    
        yb[:,0] = self.a  * yb[:,0]
        yb[:,1] = self.b
        yb[:,2] = self.T  * yb[:,2]
        yb.requires_grad = True
        
        t0 = torch.rand(n_boundary_each, 3, dtype=torch.float32)    
        t0[:,0] = self.a * t0[:,0]
        t0[:,1] = self.b * t0[:,1]
        t0[:,2] = 0
        t0.requires_grad = True 
        return (x_interior_tensor, x0, xa, y0, yb, t0)
    
    
    def construct_train_dataloader(self, train_batch_size=32, n_samples=1000):
        x_interior_tensor, x0, xa, y0, yb, t0 = self.sample_data_points(n_samples, n_samples)
        x_tensor = torch.concat((x_interior_tensor, x0, xa, y0, yb, t0), dim=1)
        dataset = UnlabeledDataset(x_tensor)
        loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)
        return loader
    
    def split_batch(self,x):
         x_interior_tensor = x[:,:3].view(-1,3)
         x0 = x[:,3:6].view(-1,3) 
         xa = x[:,6:9].view(-1,3)
         y0 = x[:,9:12].view(-1,3)
         yb = x[:,12:15].view(-1,3) 
         t0 = x[:,15:18].view(-1,3)
         return x_interior_tensor, x0, xa, y0, yb, t0
        
    def generate_grid(self, n_each_axis=100):
        x = np.linspace(0, self.a, n_each_axis)
        y = np.linspace(0, self.b, n_each_axis)
        t = np.linspace(0, self.T, n_each_axis)
        Y, T, X = np.meshgrid(y, t, x)
        XYT = np.column_stack((X.flatten(), Y.flatten(), T.flatten()))
        XYT_tensor = torch.tensor(XYT, dtype=torch.float32)
        return XYT_tensor
        

    ## Evaluate Loss
    def ResidualLoss(self, xs, u_hats):
        
        x, x0, xa, y0, yb, t0 =  xs
        u_hat, u_hat_x0, u_hat_xa, u_hat_y0, u_hat_yb, u_hat_t0 = u_hats
        
        Du_hat  = torch.autograd.grad(u_hat, x,
                                      grad_outputs=torch.ones_like(u_hat), 
                                      create_graph=True)[0]      # 1st derivative, [ batch_size , 3 ]

        Dxxu_hat = torch.autograd.grad(Du_hat[:,0], x,
                                       grad_outputs=torch.ones_like(Du_hat[:,0]),
                                       create_graph=True)[0][:,0]  # 2nd derivative, [ batch_size ,  ]
                
        Dyyu_hat = torch.autograd.grad(Du_hat[:,1], x,
                                       grad_outputs=torch.ones_like(Du_hat[:,1]),
                                       create_graph=True)[0][:,1]  # 2nd derivative, [ batch_size ,  ]
        
        Dttu_hat = torch.autograd.grad(Du_hat[:,2], x,
                                       grad_outputs=torch.ones_like(Du_hat[:,2]),
                                       create_graph=True)[0][:,2]  # 2nd derivative, [ batch_size ,  ]
        
        Lp = (((Dxxu_hat + Dyyu_hat)*self.c**2 - Dttu_hat)**2).mean()
        
        Lb1 = (u_hat_x0**2).mean()
        Lb2 = (u_hat_xa**2).mean()
        Lb3 = (u_hat_y0**2).mean()
        Lb4 = (u_hat_yb**2).mean()
        
        Du_hat_t0  = torch.autograd.grad(u_hat_t0, t0,
                                         grad_outputs=torch.ones_like(u_hat_t0), 
                                         create_graph=True)[0][:,2]  # 1st derivative, [ batch_size, ]
 
        LI1 = ((u_hat_t0 - self.g(t0))**2).mean()
        LI2 = ((Du_hat_t0 - self.phi(t0))**2).mean()
        
        loss =  (self.lambdas[0]*Lp + 
                self.lambdas[1]*Lb1 + self.lambdas[2]*Lb2 + self.lambdas[3]*Lb3 + self.lambdas[4]*Lb4 + 
                self.lambdas[5]*LI1 + self.lambdas[6]*LI2)

        return loss
    
    
    def L2_error(self, true_sol, n_each_axis=100):
        self.eval()
        XYT = self.generate_grid(n_each_axis)
        X = XYT[:,0].reshape(n_each_axis,n_each_axis,n_each_axis)[0][0,:]
        Y = XYT[:,1].reshape(n_each_axis,n_each_axis,n_each_axis)[0][:,0]
        T = XYT[:,2].reshape(n_each_axis,n_each_axis,n_each_axis)[:,1,1]
        
        U = self.forward(XYT)
        U_true = true_sol(XYT)
        SqError = (U-U_true)**2
        SqError = SqError.reshape(n_each_axis,n_each_axis,n_each_axis)
        
        t_slice = []
        for k in range(len(T)):
            slice = []
            for i in range(len(X)) :
                Y_slice =  torch.trapz(SqError[k,:,i], Y)
                slice.append(Y_slice.item())
            res = torch.trapz(torch.tensor(slice, dtype=torch.float32), X)
            t_slice.append(res)
        res = torch.trapz(torch.tensor(t_slice, dtype=torch.float32), T) 
        return (res.item())**0.5
    
    
    ## Forward pass function
    def forward(self, x): 
        u = self.fwd(x) 
        return u 
    
    ## Validation function
    def Validate(self, sample_num):
        self.eval()
        xs = self.sample_data_points(n_interior=sample_num, 
                                     n_boundary_each=sample_num)
        u_hats = []
        for x in xs:
            u_hats.append(self.forward(x))
        loss = self.ResidualLoss(xs, tuple(u_hats))      
        return loss.item()
    
    ## Test function
    def Test(self, sample_num):
        test_loss = self.Validate(sample_num)
        print( 'Test set: Avg. Test Sample Loss: {:.4f}'.format(test_loss) )
        return test_loss
    
    
    ## Training function
    def Train(self, train_num, train_batch_size=32, learning_rate=0.001,  
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
                xs = self.split_batch(x)
                u_hats = []
                for x in xs:
                    u_hats.append(self.forward(x))
                u_hats = tuple(u_hats)
                loss = self.ResidualLoss(xs, u_hats)       
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