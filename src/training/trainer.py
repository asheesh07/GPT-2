import torch
import torch.nn as nn
import math
from train_utils import compute_loss
class Trainer(nn.Module):
    def __init__(self,model,optimizer,scheduler,train_loader,val_loader,config):
        self.model=model.to(config.device)
        self.optimizer=optimizer
        self.scheduler=scheduler
        self.train_loader=train_loader
        self.val_loader=val_loader
        self.cnf=config
        self.device=config.device
        self.scaler=torch.cuda.amp.GradScaler(enabled=config.use_amp)
        self.global_step=0
        self.epoch=0
    def train_epochs(self):
        self.model.train()
        for batch in self.train_loader:
            loss=self.train_epoch_step(batch)
            self.global_step+=1
            
    def train_epoch_step(self,batch):
        batch={k:v.device(self.device) for k,v in batch.items()}
        
        with torch.cuda.amp.autocast(enabled=self.config.use_amp):
            logits=self.model(batch['input_ids'])
            loss=compute_loss(logits,batch['labels'])
            
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        if self.scheduler is not None:
            self.scheduler.step()
            
        return loss.item()
    def validate(self):
        self.model.eval()
        losses=[]
        with torch.no_grad():
            for batch in self.val_loader:
                logits=self.model(batch['input_ids'])
                loss=compute_loss(logits,batch['labels'])
                losses.append(loss.item())
        avg_loss = sum(losses) / len(losses)
        perplexity = math.exp(avg_loss)

        return {
            "val_loss": avg_loss,
            "val_ppl": perplexity
            }
    
    def check_points(self,path):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "step": self.global_step,
            "epoch": self.epoch
        }, path)  

        
            
