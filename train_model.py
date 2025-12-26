import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import json
from model import DrugTargetInteractionNN
from data_loader import create_dataloaders
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class ModelTrainer:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.writer = SummaryWriter('runs/drug_target_dti')
    
    def train_epoch(self, train_loader, criterion, optimizer):
        self.model.train()
        total_loss = 0
        for drug_feat, prot_feat, labels in train_loader:
            drug_feat = drug_feat.to(self.device)
            prot_feat = prot_feat.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(drug_feat, prot_feat)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)
    
    def validate(self, val_loader, criterion):
        self.model.eval()
        total_loss = 0
        all_preds, all_probs, all_labels = [], [], []
        with torch.no_grad():
            for drug_feat, prot_feat, labels in val_loader:
                drug_feat = drug_feat.to(self.device)
                prot_feat = prot_feat.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(drug_feat, prot_feat)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                probs = outputs.cpu().numpy()
                preds = (probs > 0.5).astype(int)
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        metrics = {
            'loss': total_loss / len(val_loader),
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds),
            'recall': recall_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds),
            'auc_roc': roc_auc_score(all_labels, all_probs),
        }
        return metrics
    
    def train(self, train_loader, val_loader, epochs=50, lr=0.001, patience=10):
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

        best_val_auc = 0
        patience_counter = 0
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            train_loss = self.train_epoch(train_loader, criterion, optimizer)
            val_metrics = self.validate(val_loader, criterion)
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['auc_roc']:.4f}")
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            self.writer.add_scalar('Metrics/val_auc', val_metrics['auc_roc'], epoch)
            self.writer.add_scalar('Metrics/val_f1', val_metrics['f1'], epoch)
            scheduler.step(val_metrics['auc_roc'])
            if val_metrics['auc_roc'] > best_val_auc:
                best_val_auc = val_metrics['auc_roc']
                patience_counter = 0
                os.makedirs('models', exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auc': best_val_auc,
                    'metrics': val_metrics
                }, 'models/best_model.pt')
                with open('models/model_config.json', 'w') as f:
                    json.dump({
                        'drug_input_dim': 2056,
                        'protein_input_dim': 3000,
                        'hidden_dims': [1024, 512, 256],
                        'dropout': 0.3
                    }, f)
                print(f"✓ Best model saved (AUC: {best_val_auc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        self.writer.close()
        print(f"\nTraining complete! Best Val AUC: {best_val_auc:.4f}")

if __name__ == '__main__':
    db_path = 'chembl_36.db'
    sample_limit = 20000
    batch_size = 128
    epochs = 50

    train_loader, val_loader, test_loader, metadata = create_dataloaders(
        db_path, sample_limit=sample_limit, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DrugTargetInteractionNN(
        drug_input_dim=metadata['drug_feat_dim'],
        protein_input_dim=metadata['protein_feat_dim']
    )

    trainer = ModelTrainer(model, device)
    trainer.train(train_loader, val_loader, epochs=epochs)
