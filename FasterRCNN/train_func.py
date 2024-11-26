import numpy as np
from tqdm import tqdm
import torch
from utils import AverageMeter

def save_history(train_loss_list, val_loss_list, save_path):
    history = {}

    history['train_loss'] = train_loss_list
    history['val_loss'] = val_loss_list

    np.save(save_path, history)

def lr_cosine_decay(base_learning_rate, global_step, decay_steps, alpha=0):
    """
    Params
        - learning_rate : Base Learning Rate
        - global_step : Current Step in Train Pipeline
        - decay_steps : Total Decay Steps in Learning Rate
        - alpha : Learning Scaled Coefficient
    """
    global_step = min(global_step, decay_steps)
    cosine_decay = 0.5 * (1 + np.cos(np.pi * global_step / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    decayed_learning_rate = base_learning_rate * decayed

    return decayed_learning_rate

def train(args, model, train_dataloader, val_dataloader, optimizer):

    # for gradient clipping
    max_norm = args.max_norm
    
    start_epoch = 0
    total_iter = len(train_dataloader) * args.epochs
    global_step = 0
    minimum_val_loss = float("inf")
    
    train_losses_avg = []
    val_losses_avg = []

    # Training
    for epoch in range(start_epoch, args.epochs):
        # loss recorder
        train_losses = AverageMeter()
        val_losses = AverageMeter()

        # model train mode
        model.train()
        train_t = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for i, (images, targets, image_id) in train_t:
            try:
                images = list(image.to('cuda') for image in images)
                targets = [{k: v.to('cuda') for k, v in t.items()} for t in targets]
    
                # calculate loss
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
    
                # update weight
                optimizer.zero_grad()
                losses.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
    
                # adjust lr
                global_step += 1
                adjust_lr = lr_cosine_decay(args.lr, global_step, total_iter)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = adjust_lr
    
                # loss update
                train_losses.update(losses.item())
    
                # print tqdm
                print_loss = round(losses.item(), 4)
                train_t.set_postfix_str("Train loss : {}".format(print_loss))
                
            except:
                print("Invalid Box Error so Ignore...")
                print(image_id)
            
        # Update learning rates schedule
        train_losses_avg.append(train_losses.avg)

        # Validation
        val_t = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
        with torch.no_grad():
            try:
                for i, (images, targets, image_id) in val_t:
                    images = list(image.to('cuda') for image in images)
                    targets = [{k: v.to('cuda') for k, v in t.items()} for t in targets]
                    
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
        
                    # loss recording
                    val_losses.update(losses.item())
        
                    # print tqdm
                    print_loss = round(losses.item(), 4)
                    val_t.set_postfix_str("Val loss : {}".format(print_loss))
            except:
                print("Invalid Box Error so Ignore...")
                print(image_id)
                
        # loss recording
        val_losses_avg.append(val_losses.avg)

        # print train loss, val loss
        print("Epoch : {}   Train Loss : {}  Val Loss : {}".format(epoch+1, round(train_losses.avg, 4), round(val_losses.avg, 4)))

        # save best model
        if args.monitor == 'loss':
            val_avg_loss = val_losses.avg
            if val_avg_loss<minimum_val_loss:
                print('improve val_loss!! so model save {} -> {}'.format(minimum_val_loss, val_avg_loss))
                minimum_val_loss = val_avg_loss
                torch.save(model.state_dict(), args.model_save_path)

        # save history
        save_history(train_losses_avg, val_losses_avg, save_path=args.model_save_path.replace('.pth', '.npy'))
