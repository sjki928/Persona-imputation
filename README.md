# Persona-imputation

AIhub관광 데이터를 이용해 여행지를 다녀온 사람들의 Presona의 분포를 학습

## Method
- 데이터 형식은 여행지에 대한 text가 Input 여행지를 다녀온 관광객들의 Presona Vectors의 mean, variance vector가 Label인 형태로 구성되어 있다.

- VAE를 modify하며 사용중이며, 이때 KL-Divergence term을 Standard Normal distribution을 기준으로 설정하지 않고, Label distribution과의 KL-Divergence로 계산하였다. 이때 multi-variate normal distribution으로 계산하기엔 inverse matrix문제가 발생하여 각 variation마다 계산하였다.

- Ablation Study를 통해 Reconstruction term이 존재하지 않는, 즉 Decoder가 없는 형태의 Neural Architecture를 구성했을 때 Overfitting과 별개로 train error가 줄어들지 않는 모습을 보여 VAE의 형태를 유지 중


## 현황
- 현재 데이터의 부족으로 1epoch만에 Overfitting 되는 모습을 보여 수작업으로 웹에서 설명 데이터를 추가하고 있었으며 현재 약 130개 가량 수집함. 아래 파일에 저장 중

```
./data/empty_tour.csv
```