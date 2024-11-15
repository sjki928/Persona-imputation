import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import VAE
from dataloader import TourDataset, build_dataframe

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    source: https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/losses.py#L12
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )

if __name__ == '__main__':
    input_dim = 384
    hidden_dim = 128
    latent_dim = 13
    lr = 1e-4
    batch_size = 8
    epochs = 20

    vae = VAE(input_dim, hidden_dim, latent_dim)
    optimizer = optim.Adam(vae.parameters(), lr=lr)


    train_dataframe, test_dataframe, scaler = build_dataframe('/workspace/vae_train_nara.csv','sentence-transformers/all-MiniLM-L6-v2')
    
    train_dataset = TourDataset(train_dataframe)
    test_dataset = TourDataset(test_dataframe)
    
    train_loader = DataLoader(train_dataset,batch_size=8,shuffle=True)
    test_loader =  DataLoader(test_dataset,batch_size=8,shuffle=True)
    
    for epoch in range(epochs):
        # ====== 학습 단계 ======
        vae.train()
        train_loss = 0
        for batch in train_loader:
            data = batch['textvec']
            mu = batch['mean']
            var = batch['var']
            
            
            optimizer.zero_grad()
            
            reconstruction, mean, logvar, var = vae(data)
            
            
            kl_loss = normal_kl(mean, logvar, mu, torch.log(var))
            kl_loss = torch.mean(kl_loss)
            
            rec_loss = torch.abs(data.contiguous() - reconstruction.contiguous())**2
            rec_loss = torch.mean(rec_loss)

            loss = kl_loss+rec_loss
                        
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        
        # ====== 검증 단계 ======
        vae.eval()
        val_loss = 0
        
        with torch.no_grad():  # 검증에서는 gradient 계산을 하지 않음
            for batch in test_loader:
                data = batch['textvec']
                mu = batch['mean']
                cov = batch['var']
                
                
                optimizer.zero_grad()
                
                reconstruction, mean, logvar, var = vae(data)
                
                
                kl_loss = normal_kl(mean, var, mu, torch.log(var))
                kl_loss = torch.mean(kl_loss)
                
                rec_loss = torch.abs(data.contiguous() - reconstruction.contiguous())**2
                rec_loss = torch.mean(rec_loss)

                loss = kl_loss+rec_loss
                                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(test_loader.dataset)
        
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")