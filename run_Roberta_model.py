import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from transformers.optimization import AdamW
from sys import platform

from data_sst2 import DataPrecessForSentence
from utils import train, validate, test
from models import RobertModel

def model_train_validate_test(train_df, dev_df, test_df, target_dir, 
         max_seq_len=128,
         epochs=10,
         batch_size=32,
         lr=2e-06,
         patience=5,
         max_grad_norm=1.0,
         if_save_model=True,
         checkpoint=None):
    """
    Parameters
    ----------
    train_df : pandas dataframe of train set.
    dev_df : pandas dataframe of dev set.
    test_df : pandas dataframe of test set.
    target_dir : the path where you want to save model.
    max_seq_len: the max truncated length.
    epochs : the default is 3.
    batch_size : the default is 32.
    lr : learning rate, the default is 2e-05.
    patience : the default is 1.
    max_grad_norm : the default is 10.0.
    if_save_model: if save the trained model to the target dir.
    checkpoint : the default is None.

    """

    bertmodel = RobertModel(requires_grad = True)
    tokenizer = bertmodel.tokenizer
    
    print(20 * "=", " Preparing for training ", 20 * "=")
    # Path to save the model, create a folder if not exist.
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    # -------------------- Data loading --------------------------------------#
    
    print("\t* Loading training data...")
    train_data = DataPrecessForSentence(tokenizer, train_df, max_seq_len = max_seq_len)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    print("\t* Loading validation data...")
    dev_data = DataPrecessForSentence(tokenizer,dev_df, max_seq_len = max_seq_len)
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=batch_size)
    
    print("\t* Loading test data...")
    test_data = DataPrecessForSentence(tokenizer,test_df, max_seq_len = max_seq_len) 
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    
    # -------------------- Model definition ------------------- --------------#
    
    print("\t* Building model...")
    device = torch.device("cuda")
    model = bertmodel.to(device)
    
    # -------------------- Preparation for training  -------------------------#
    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {
                    'params':[p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                    'weight_decay':0.01
            },
            {
                    'params':[p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                    'weight_decay':0.0
            }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    ## Implement of warm up
    ## total_steps = len(train_loader) * epochs
    ## scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=60, num_training_steps=total_steps)
    
    # When the monitored value is not improving, the network performance could be improved by reducing the learning rate.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.85, patience=0)

    best_score = 0.0
    start_epoch = 1
    # Data for loss curves plot
    epochs_count = []
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []
    valid_aucs = []
    
    # Continuing training from a checkpoint if one was given as argument
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]
        print("\t* Training will continue on existing model from epoch {}...".format(start_epoch))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epochs_count = checkpoint["epochs_count"]
        train_losses = checkpoint["train_losses"]
        train_accuracy = checkpoint["train_accuracy"]
        valid_losses = checkpoint["valid_losses"]
        valid_accuracy = checkpoint["valid_accuracy"]
        valid_auc = checkpoint["valid_auc"]
     # Compute loss and accuracy before starting (or resuming) training.
    _, valid_loss, valid_accuracy, _ = validate(model, dev_loader)
    print("\n* Validation loss before training: {:.4f}, accuracy: {:.4f}%".format(valid_loss, (valid_accuracy*100)))
    
    # -------------------- Training epochs -----------------------------------#
    
    print("\n", 20 * "=", "Training bert model on device: {}".format(device), 20 * "=")
    patience_counter = 0
    for epoch in range(start_epoch, epochs + 1):
        epochs_count.append(epoch)

        print("* Training epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = train(model, train_loader, optimizer, epoch, max_grad_norm)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)  
        print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%".format(epoch_time, epoch_loss, (epoch_accuracy*100)))
        
        print("* Validation for epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy, _ = validate(model, dev_loader)
        valid_losses.append(epoch_loss)
        valid_accuracies.append(epoch_accuracy)
        print("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%"
              .format(epoch_time, epoch_loss, (epoch_accuracy*100)))
        
        # Update the optimizer's learning rate with the scheduler.
        scheduler.step(epoch_accuracy)
        ## scheduler.step()
        
        # Early stopping on validation accuracy.
        if epoch_accuracy < best_score:
            patience_counter += 1
        else:
            best_score = epoch_accuracy
            patience_counter = 0
            if (if_save_model):
                  torch.save({"epoch": epoch, 
                           "model": model.state_dict(),
                           "optimizer": optimizer.state_dict(),
                           "best_score": best_score,
                           "epochs_count": epochs_count,
                           "train_losses": train_losses,
                           "train_accuracy": train_accuracies,
                           "valid_losses": valid_losses,
                           "valid_accuracy": valid_accuracies,
                           "valid_auc": valid_aucs
                           },
                           os.path.join(target_dir, "best.pth.tar"))
                  print("save model succesfully!\n")
            
            # run model on test set and save the prediction result to csv
            print("* Test for epoch {}:".format(epoch))
            _, _, test_accuracy, all_prob = validate(model, test_loader)
            print("Test accuracy: {:.4f}%\n".format(test_accuracy))
            test_prediction = pd.DataFrame({'prob_1':all_prob})
            test_prediction['prob_0'] = 1-test_prediction['prob_1']
            test_prediction['prediction'] = test_prediction.apply(lambda x: 0 if (x['prob_0'] > x['prob_1']) else 1, axis=1)
            test_prediction = test_prediction[['prob_0', 'prob_1', 'prediction']]
            test_prediction.to_csv(os.path.join(target_dir,"test_prediction.csv"), index=False)
             
        if patience_counter >= patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break


def model_load_test(test_df, target_dir, test_prediction_dir, test_prediction_name, max_seq_len=50, batch_size=32):
    """
    Parameters
    ----------
    test_df : pandas dataframe of test set.
    target_dir : the path of pretrained model.
    test_prediction_dir : the path that you want to save the prediction result to.
    test_prediction_name : the file name of the prediction result.
    max_seq_len: the max truncated length.
    batch_size : the default is 32.
    
    """
    bertmodel = RobertModel(requires_grad = False)
    tokenizer = bertmodel.tokenizer
    device = torch.device("cuda")
    
    print(20 * "=", " Preparing for testing ", 20 * "=")
    if platform == "linux" or platform == "linux2":
        checkpoint = torch.load(os.path.join(target_dir, "best.pth.tar"))
    else:
        checkpoint = torch.load(os.path.join(target_dir, "best.pth.tar"), map_location=device)
        
    print("\t* Loading test data...")    
    test_data = DataPrecessForSentence(tokenizer,test_df, max_seq_len = max_seq_len) 
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    # Retrieving model parameters from checkpoint.
    print("\t* Building model...")
    model = bertmodel.to(device)
    model.load_state_dict(checkpoint["model"])
    print(20 * "=", " Testing BERT model on device: {} ".format(device), 20 * "=")
    
    batch_time, total_time, accuracy, all_prob = test(model, test_loader)
    print("\n-> Average batch processing time: {:.4f}s, total test time: {:.4f}s, accuracy: {:.4f}%\n".format(batch_time, total_time, (accuracy*100)))
    
    test_prediction = pd.DataFrame({'prob_1':all_prob})
    test_prediction['prob_0'] = 1-test_prediction['prob_1']
    test_prediction['prediction'] = test_prediction.apply(lambda x: 0 if (x['prob_0'] > x['prob_1']) else 1, axis=1)
    test_prediction = test_prediction[['prob_0', 'prob_1', 'prediction']]
    if not os.path.exists(test_prediction_dir):
        os.makedirs(test_prediction_dir)
    test_prediction.to_csv(os.path.join(test_prediction_dir, test_prediction_name), index=False)


if __name__ == "__main__":
    data_path = "./data/financial_phrasebank/"
    train_df = pd.read_csv(os.path.join(data_path,"combined_train.csv"))
    print(train_df.head())
    train_df = train_df[['label','sentence']]
    train_df.columns = ['similarity','s1']
    dev_df = pd.read_csv(os.path.join(data_path,"combined_dev.csv"))
    dev_df = dev_df[['label','sentence']]
    dev_df.columns = ['similarity','s1']
    test_df = pd.read_csv(os.path.join(data_path,"combined_test.csv"))
    test_df = test_df[['label','sentence']]
    test_df.columns = ['similarity','s1']
    target_dir = "./data/financial_phrasebank//output/Roberta/"
    model_train_validate_test(train_df, dev_df, test_df, target_dir)