"""
This script trains and evaluates a mammography classification model using a Vision Transformer with Mixture-of-Experts.
It supports logging via wandb, uses focal loss to handle class imbalance, and saves checkpoints during training.
"""

import os
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from PIL import Image
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR

from torchvision import transforms

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import sys
import wandb

from models.models_m4oe import SoftMoEVisionTransformer
from datasets.EMBED import MammographyDataset_complete


class Logger(object):
    """
    Logger class that duplicates stdout messages to both the console and a log file.
    """
    def __init__(self, filename="log.txt"):
        """
        Initialize the Logger.

        Args:
            filename (str): File path to save the log.
        """
        self.terminal = sys.stdout  # Save the original standard output.
        self.log = open(filename, "w", encoding="utf-8")  # Open log file for writing.

    def write(self, message):
        """
        Write a message to both the console and the log file.

        Args:
            message (str): Message to write.
        """
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        """
        Flush the output buffers of both console and log file.
        """
        self.terminal.flush()
        self.log.flush()


class FocalLoss(nn.Module):
    """
    Focal Loss module that down-weights well-classified examples to focus training on hard negatives.

    Attributes:
        gamma (float): Focusing parameter that adjusts the rate at which easy examples are down-weighted.
        alpha (Tensor or None): Weighting factor for classes.
        size_average (bool): Determines if the loss is averaged over the batch.
    """
    def __init__(self, gamma=0, alpha=None, size_average=True):
        """
        Initialize the FocalLoss.

        Args:
            gamma (float): Focusing parameter gamma.
            alpha (Tensor, list, or None): Weighting factors for classes.
            size_average (bool): If True, average the loss over all samples.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        """
        Compute the focal loss between the input and target.

        Args:
            input (Tensor): Predicted logits with shape [N, C, ...].
            target (Tensor): Ground truth labels with shape [N, ...].

        Returns:
            Tensor: Computed focal loss.
        """
        # If input has more than 2 dimensions, reshape it to (N*H*W, C)
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # (N, C, H*W)
            input = input.transpose(1, 2)  # (N, H*W, C)
            input = input.contiguous().view(-1, input.size(2))  # (N*H*W, C)
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)  # Compute log-softmax
        logpt = logpt.gather(1, target)  # Gather log-probabilities corresponding to target labels
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())  # Convert log-probabilities to probabilities

        if self.alpha is not None:
            # Ensure alpha is of same type as input data
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        # Compute the focal loss
        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def train(model, train_loader, test_loader, 
          criterion_1, criterion_3, criterion_2, 
          optimizer, scheduler, num_epochs, use_wandb, save_dir):
    """
    Train the model and periodically evaluate on a test set.

    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        test_loader (DataLoader): DataLoader for the test dataset.
        criterion_1 (nn.Module): Loss function for the first label.
        criterion_3 (nn.Module): Loss function for the third label.
        criterion_2 (nn.Module): Loss function for the second label.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        num_epochs (int): Number of training epochs.
        use_wandb (bool): Flag to enable wandb logging.
        save_dir (str): Directory to save checkpoints.
    """
    global_step = 0
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode.
        running_loss = 0.0

        # Lists to collect predictions and true labels for performance metrics.
        all_preds1, all_labels1 = [], []
        all_preds3, all_labels3 = [], []
        all_preds2, all_labels2 = [], []

        # Iterate over batches in the training set.
        for i, ((c_view_cc, c_view_mlo, d2_cc, d2_mlo), 
                (label1, label3, label2), record_id) in enumerate(tqdm(train_loader, total=len(train_loader))):
            global_step += 1

            # Move inputs and labels to GPU.
            c_view_cc = c_view_cc.to('cuda')
            c_view_mlo = c_view_mlo.to('cuda')
            d2_cc = d2_cc.to('cuda')
            d2_mlo = d2_mlo.to('cuda')
            label1 = label1.long().to('cuda')
            label3 = label3.long().to('cuda')
            label2 = label2.long().to('cuda')

            optimizer.zero_grad()  # Zero the gradients.

            # Forward pass: obtain outputs and an additional loss component (tc_loss)
            outputs1, outputs3, outputs2, tc_loss = model((c_view_cc, c_view_mlo, d2_cc, d2_mlo))
            
            # Create a mask for valid labels (non-negative)
            valid_idx = (label2 >= 0) & (label1 >= 0) & (label3 >= 0)
            if valid_idx.sum() > 0:
                loss2 = criterion_2(outputs2[valid_idx], label2[valid_idx])
                loss1 = criterion_1(outputs1[valid_idx], label1[valid_idx])
                loss3 = criterion_3(outputs3[valid_idx], label3[valid_idx])
            else:
                loss2 = torch.tensor(0.0, device='cuda')
                loss1 = torch.tensor(0.0, device='cuda')
                loss3 = torch.tensor(0.0, device='cuda')

            # Combine losses (optionally include tc_loss if needed)
            # loss = loss1 + loss2 + loss3 + 0.05 * tc_loss  # UNCOMMENT WHEN NEEDED
            loss = loss1 + loss2 + loss3  # UNCOMMENT WHEN NEEDED
            
            # TODO CODE CLEANING FOR NEXT RELEASE : 
            # RELEASE ADAPTED AUTO MULTI-LOSS REWEIGHTING MECHANISM 
            # FROM Mod-Squad: Designing Mixtures of Experts As Modular Multi-Task Learners
            
            loss.backward()  # Backpropagation.
            optimizer.step()  # Update parameters.
            running_loss += loss.item()

            # Get predicted classes from model outputs.
            _, pred1 = torch.max(outputs1.data, 1)
            _, pred3 = torch.max(outputs3.data, 1)
            _, pred2 = torch.max(outputs2.data, 1)

            # Store predictions and labels for later evaluation.
            all_preds1.extend(pred1.cpu().numpy())
            all_labels1.extend(label1.cpu().numpy())
            all_preds3.extend(pred3.cpu().numpy())
            all_labels3.extend(label3.cpu().numpy())

            if valid_idx.sum() > 0:
                valid_pred2 = pred2[valid_idx]
                valid_label2 = label2[valid_idx]
                all_preds2.extend(valid_pred2.cpu().numpy())
                all_labels2.extend(valid_label2.cpu().numpy())

            # Every 2000 global steps, evaluate on the test set and save a checkpoint.
            if global_step % 2000 == 0:
                print(f"\n*** Global step {global_step}: Running test ***")
                test(model, test_loader, use_wandb)
                checkpoint_path = os.path.join(save_dir, f"checkpoint_step_{global_step}.pt")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}\n")

                # Compute and print metrics.
                acc1 = accuracy_score(all_labels1, all_preds1)
                f1_1 = f1_score(all_labels1, all_preds1, average='macro')
                acc3 = accuracy_score(all_labels3, all_preds3)
                f1_3 = f1_score(all_labels3, all_preds3, average='macro')
                acc2 = accuracy_score(all_labels2, all_preds2)
                f1_2 = f1_score(all_labels2, all_preds2, average='macro')
                print(f'Epoch {epoch+1} - Label1: Accuracy: {acc1:.3f}, F1: {f1_1:.3f}')
                print(f'Epoch {epoch+1} - Label3: Accuracy: {acc3:.3f}, F1: {f1_3:.3f}')
                print(f'Epoch {epoch+1} - Label2: Accuracy: {acc2:.3f}, F1: {f1_2:.3f}')

                # Reset the prediction/label lists after evaluation.
                if global_step > 0:
                    all_preds1, all_labels1 = [], []
                    all_preds3, all_labels3 = [], []
                    all_preds2, all_labels2 = [], []

            # Log average loss every 10 batches.
            if i % 10 == 9:
                avg_loss = running_loss / 100
                print(f'[Epoch {epoch+1}, Batch {i+1}] loss: {avg_loss:.3f}')
                if use_wandb:
                    wandb.log({"train_loss": avg_loss})
                running_loss = 0.0

        # Update the learning rate scheduler at the end of the epoch.
        scheduler.step()

        # Log epoch-level metrics to wandb if enabled.
        if use_wandb:
            wandb.log({
                "train_accuracy_label1": acc1, "train_f1_label1": f1_1,
                "train_accuracy_label3": acc3, "train_f1_label3": f1_3,
                "train_accuracy_label2": acc2, "train_f1_label2": f1_2
            })
        # Save the model checkpoint after each epoch.
        torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch+1}.pt"))


def test(model, test_loader, use_wandb):
    """
    Evaluate the model on the test set and report accuracy, F1 score, and confusion matrix for each label.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test dataset.
        use_wandb (bool): Flag to enable wandb logging.

    Returns:
        tuple: Three tuples containing (accuracy, F1 score, confusion matrix) for label1, label3, and label2.
    """
    model.eval()  # Set model to evaluation mode.
    total = 0
    correct1, correct3, correct2 = 0, 0, 0
    all_test_preds1, all_test_preds3 = [], []
    all_test_preds2 = []
    all_test_labels1, all_test_labels3 = [], []
    all_test_labels2 = []

    with torch.no_grad():
        # Iterate over batches in the test dataset.
        for ((c_view_cc, c_view_mlo, d2_cc, d2_mlo), 
             (label1, label3, label2), record_id) in test_loader:
            # Move data to GPU.
            c_view_cc = c_view_cc.to('cuda')
            c_view_mlo = c_view_mlo.to('cuda')
            d2_cc = d2_cc.to('cuda')
            d2_mlo = d2_mlo.to('cuda')
            label1 = label1.long().to('cuda')
            label3 = label3.long().to('cuda')
            label2 = label2.long().to('cuda')
            
            # Forward pass to get outputs.
            outputs1, outputs3, outputs2, tc_loss = model((c_view_cc, c_view_mlo, d2_cc, d2_mlo))
            _, pred1 = torch.max(outputs1.data, 1)
            _, pred3 = torch.max(outputs3.data, 1)
            _, pred2 = torch.max(outputs2.data, 1)
            
            total += label1.size(0)
            
            # Create masks for valid labels.
            valid_mask2 = label2 >= 0
            if valid_mask2.sum() > 0:
                correct2 += (pred2[valid_mask2] == label2[valid_mask2]).sum().item()
            valid_mask3 = label3 >= 0
            if valid_mask3.sum() > 0:
                correct3 += (pred3[valid_mask3] == label3[valid_mask3]).sum().item()
            valid_mask1 = label1 >= 0
            if valid_mask1.sum() > 0:
                correct1 += (pred1[valid_mask1] == label1[valid_mask1]).sum().item()
            
            # Collect predictions and labels for metric computation.
            if valid_mask2.sum() > 0:
                all_test_preds2.extend(pred2[valid_mask2].cpu().numpy())
                all_test_labels2.extend(label2[valid_mask2].cpu().numpy())
            if valid_mask3.sum() > 0:
                all_test_preds3.extend(pred3[valid_mask3].cpu().numpy())
                all_test_labels3.extend(label3[valid_mask3].cpu().numpy())
            if valid_mask1.sum() > 0:
                all_test_preds1.extend(pred1[valid_mask1].cpu().numpy())
                all_test_labels1.extend(label1[valid_mask1].cpu().numpy())
    
    # Calculate accuracy for each label.
    accuracy_val1 = 100 * correct1 / len(all_test_preds1)
    accuracy_val3 = 100 * correct3 / len(all_test_preds3)
    accuracy_val2 = 100 * correct2 / len(all_test_preds2)
    
    # Compute F1 scores.
    f1_val1 = f1_score(all_test_labels1, all_test_preds1, average='macro')
    f1_val3 = f1_score(all_test_labels3, all_test_preds3, average='macro')
    f1_val2 = f1_score(all_test_labels2, all_test_preds2, average='macro')
    
    # Compute confusion matrices.
    cm1 = confusion_matrix(all_test_labels1, all_test_preds1)
    cm3 = confusion_matrix(all_test_labels3, all_test_preds3)
    cm2 = confusion_matrix(all_test_labels2, all_test_preds2)
    
    # Print the evaluation metrics.
    print(f'Test Accuracy for Label1: {accuracy_val1:.2f}%')
    print(f'Test F1 for Label1: {f1_val1:.3f}')
    print(f'Confusion Matrix for Label1:\n{cm1}')
    print(f'Test Accuracy for Label3: {accuracy_val3:.2f}%')
    print(f'Test F1 for Label3: {f1_val3:.3f}')
    print(f'Confusion Matrix for Label3:\n{cm3}')
    print(f'Test Accuracy for Label2: {accuracy_val2:.2f}%')
    print(f'Test F1 for Label2: {f1_val2:.3f}')
    print(f'Confusion Matrix for Label2:\n{cm2}')
    
    return ((accuracy_val1, f1_val1, cm1),
            (accuracy_val3, f1_val3, cm3),
            (accuracy_val2, f1_val2, cm2))


def main():
    """
    Parse command-line arguments, set up data loaders, model, loss functions, optimizer, and scheduler,
    and then start the training and evaluation process.
    """
    parser = argparse.ArgumentParser(description='Train a single-task mammography model with integrated data cleaning')
    parser.add_argument('--save_dir', type=str, default='/scratch/liyues_root/liyues/tangzx/ICLR/checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='Weight decay for optimizer')
    parser.add_argument('--use_wandb', type=bool, default=False, help='Flag to use wandb logging')
    parser.add_argument('--wandb_project', type=str, default='iclr_rebuttal_test', help='W&B project name')
    parser.add_argument('--wandb_name', type=str, default='single_task', help='W&B run name')
    parser.add_argument('--target_label', type=str, default='birads',
                        help="Which label to use. For example, 'tissueden' or 'cancer_risk_label'")
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'],
                        help="Dataset split to use: 'train' or 'test'")
    args = parser.parse_args()

    # Redirect standard output to both console and a log file.
    sys.stdout = Logger(f"ICLR_log_{args.lr}_{args.weight_decay}_{args.batch_size}_{args.seed}.txt")
    
    # Initialize wandb if enabled.
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            config={
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "optimizer": "AdamW",
                "scheduler": "StepLR",
                "seed": 42,
            },
            name=args.wandb_name
        )

    # Create the directory to save checkpoints if it does not exist.
    save_path = os.path.join(args.save_dir, f"ICLR_{args.batch_size}_{args.lr}_{args.seed}")
    os.makedirs(save_path, exist_ok=True)

    # Set random seeds for reproducibility.
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Define image transforms.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Initialize the training and testing datasets.
    train_dataset = MammographyDataset_complete(
                                       transform=transform,
                                       split='train', random_state=args.seed)
    test_dataset = MammographyDataset_complete(
                                      transform=transform,
                                      split='test', random_state=args.seed)
    
    # Create DataLoaders for training and testing.
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # Retrieve the number of classes for each label.
    num_classes_1 = train_dataset.num_classes_1  
    num_classes_2 = train_dataset.num_classes_2  
    num_classes_3 = train_dataset.num_classes_3  

    # Initialize the model with the specified parameters.
    model = SoftMoEVisionTransformer(
        num_experts=128,
        slots_per_expert=1,
        moe_layer_index=0,
        img_size=224,
        patch_size=16,
        num_classes_1=num_classes_1,
        num_classes_2=num_classes_2,
        num_classes_3=num_classes_3,
        embed_dim=768,
        depth=2,
        num_heads=8,
        mlp_ratio=4,
    )
    model = model.to('cuda')
    
    # Define class distributions for the labels.
    class_dis_1 = np.array([#REPLACE WITH THE CLASS DISTRIBUTION])  # Density distribution.
    class_weights_1 = 1 - class_dis_1 / np.sum(class_dis_1)
    class_dis_2 = np.array([#REPLACE WITH THE CLASS DISTRIBUTION])  # BIRADS distribution.
    class_weights_2 = 1 - class_dis_2 / np.sum(class_dis_2)
    class_dis_3 = np.array([#REPLACE WITH THE CLASS DISTRIBUTION])  # Risk distribution.
    class_weights_3 = 1 - class_dis_3 / np.sum(class_dis_3)
    
    # Initialize loss functions using FocalLoss for class imbalance.
    criterion_1 = FocalLoss(gamma=2, alpha=torch.tensor(class_weights_1).float())
    criterion_2 = FocalLoss(gamma=2, alpha=torch.tensor(class_weights_2).float())
    criterion_3 = FocalLoss(gamma=2, alpha=torch.tensor(class_weights_3).float())
    criterion_4 = FocalLoss(gamma=2, alpha=torch.tensor(class_weights_3).float())
    criterion_5 = FocalLoss(gamma=2, alpha=torch.tensor(class_weights_3).float())
    criterion_6 = FocalLoss(gamma=2, alpha=torch.tensor(class_weights_3).float())
    criterion_7 = FocalLoss(gamma=2, alpha=torch.tensor(class_weights_3).float())
    
    # Set up the optimizer and learning rate scheduler.
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # Begin training.
    train(model, train_loader, test_loader, criterion_1, criterion_3, 
          criterion_2, 
          optimizer, scheduler, num_epochs=args.epochs,
          use_wandb=args.use_wandb, save_dir=save_path)

    # Run final evaluation on the test set.
    test(model, test_loader, use_wandb=args.use_wandb)

    # Finish the wandb run if enabled.
    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
