import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

class GridbasedFNN(nn.Module): 
    
    ## Initialize 
    def __init__(self, loader, output_activation='Sigmoid', drop=0.0):
        super().__init__()
        self.loader = loader
        
        self.train_loss = [] 
        self.validate_loss = []
        self.loss_func = nn.MSELoss(reduction='sum')     # loss function, mse
        self.hidden1 = nn.Sequential( nn.Linear(12,40), nn.ELU() )
        self.hidden2 = nn.Sequential( nn.Linear(40,40), nn.ELU() )
        self.dropout = nn.Dropout(p=drop) 
        
        if output_activation == 'ReLU':
            self.output  = nn.Sequential( nn.Linear(40,50), nn.ReLU() )
        else:
            self.output  = nn.Sequential( nn.Linear(40,50), nn.Sigmoid() )
      

    ## Forward function
    def forward(self, x):          
        x = self.hidden1(x)      # x : [batch_size, 12] ---> [batch_size, 40] , activation
        x = self.hidden2(x)      # x : [batch_size, 40] ---> [batch_size, 40] , activation
        x = self.dropout(x)
        x = self.hidden2(x)      # x : [batch_size, 40] ---> [batch_size, 40] , activation
        x = self.dropout(x)
        x = self.hidden2(x)      # x : [batch_size, 40] ---> [batch_size, 40] , activation
        x = self.output(x)       # x : [batch_size, 40] ---> [batch_size, 50]
        return x
    
    
    ## Validation function
    def Validate(self):
            
        self.eval()                                                # convert to test mode
        validate_loss = 0 
        validate_loader = self.loader['validate']
        batch_num = len(validate_loader)                            # number of batches
        
        with torch.no_grad():    
            for x , y in validate_loader:
                
                y_hat = self.forward(x)                             # forward calculation                               
                validate_loss += self.loss_func(y_hat, y).item()    # compute the total loss for all batches in validate set
            
            validate_loss /= batch_num                    # compute average loss for each batch
            
        return validate_loss
            

    
    ## Train function
    def Train(self, num_epochs, learning_rate, lr_step_size=10, min_lr =5e-4, abs_tolerance=0.001):
        
        train_loader  = self.loader['train']
        n_train_batches = len(train_loader)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)        # optimizer, adam optimizer
        scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=0.5)   # learning updater
        
        for epoch in range(num_epochs):                # training starts
            
            print('------------------------------------------------------- ')
            print('-------------------- Epoch [{}/{}] -------------------- '.format(epoch + 1, num_epochs))
                
            self.train()                               # convert back to train mode
            train_loss = 0
            
            for i, (x,y) in enumerate(train_loader):    

                y_hat = self.forward(x)                # forward calculation
                loss = self.loss_func(y_hat, y)        # evaluate loss

                optimizer.zero_grad()                  # clear gradients
                loss.backward()                        # back propgation 
                optimizer.step()                       # update parameters
                train_loss += loss.item()              # compute the total loss for all batches in train set
                
                # Display the training progress
                if (i==0) or ((i+1) % round(n_train_batches/5) == 0):
                    print( 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                        epoch + 1, num_epochs, i + 1, n_train_batches, loss.item()) )
            
            self.validate_loss.append(self.Validate())               # record average validate batch loss for each epoch  
            self.train_loss.append(train_loss/len(train_loader))     # record average train batch loss for each epoch    
            
            print( 'Epoch [{}/{}], Avg. Train Loss: {:.4f}, Avg. Validate Loss: {:.4f}'.format(
                        epoch + 1, num_epochs, self.train_loss[-1], self.validate_loss[-1]) )
            
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
    def Test(self, test_loader):
        
        self.eval()  
        
        sample_num = len(test_loader.dataset) # number of test samples
        batch_num = len(test_loader)          # number of batches
        test_loss = 0         

        with torch.no_grad():    
            for x , y in test_loader:
                
                y_hat = self.forward(x)                                   
                test_loss += self.loss_func(y_hat, y).item()    # total loss for all batches in test set

            av_sampleloss = test_loss / sample_num
            av_batchloss  = test_loss / batch_num
           
            print( 'Test set: Avg. Sample Loss: {:.4f}, Avg. Batch Loss: {:.4f}'.format(
                av_sampleloss, av_batchloss) )
            
        return av_sampleloss , av_batchloss