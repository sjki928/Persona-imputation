import torch
import torch.nn as nn
import torch.optim as optim

# VAE 모델 클래스 정의
class VAE(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=128, latent_dim=13):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        # 인코더 정의: 384차원 입력 -> 128차원 -> 평균 및 로그 분산 벡터 (13차원)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Tanh()
        )
        
        # 잠재 공간의 평균 및 분산 벡터
        self.fc_mu = nn.Sequential(
            nn.Linear(self.hidden_dim, self.latent_dim),
            nn.Tanh()
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(self.hidden_dim, self.latent_dim),
            nn.Tanh()
        )
        
        # 디코더 정의: 13차원 잠재 공간 -> 128차원 -> 384차원 출력
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.input_dim),
            nn.Tanh()  
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
            
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, torch.exp(logvar)


if __name__ == '__main__':
    # 모델 및 최적화 설정
    input_dim = 384
    hidden_dim = 128
    latent_dim = 13

    sample = torch.randn(2,384)

    vae = VAE(input_dim, hidden_dim, latent_dim)
    
    output = vae(sample)
    print(1)
    