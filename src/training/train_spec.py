from src.training.dataset import Train_dataset

class GPT2_spec(Trainingspec):
    def __init__(self,model,train_token_path,val_token_path,block_size):
        self.model=model
        self.train_token_path=train_token_path
        self.val_token_path=val_token_path
        self.block_size=block_size
    
    def prepare_datasets(self):
        train_dataset=Train_dataset(self.train_token_path,self.block_size)
        val_dataset=train_dataset(self.val_token_path,self.block_size)
        
        return (train_dataset,val_dataset)
    def construct_model(self):
        return self.model
    
    def create_optimizer(self,params):
        opt=build_optimizer(self.model,params)
        sch=build_scheduler(opt)
        
        return opt,sch
    
    def train_objectives(self,data,model):
        logits=model(data['input_ids'])
        loss=compute_loss(logits,data['labels'])
        
        return loss.item()
    def val_objective(self,data,model):
        logits = model(data["input_ids"])
        loss = compute_loss(logits, data["labels"])
        return {"loss": loss, "logits": logits, "labels": data["labels"]}