import torch
import torch.nn as nn
import torchvision.transforms as T

class BatchOcclusion(nn.Module):
    def __init__(self, p=0.5, scale=(0.02, 0.4), ratio=(0.3, 3.3), value='random'):
        """
        p: 執行遮蔽的機率 (0.5 代表一半的圖片會被遮)
        scale: 遮蔽面積佔整張圖的比例範圍 (2% ~ 20%)
        ratio: 遮蔽長方形的長寬比範圍
        value: 遮蔽區域的填充值 (0 為黑色, 'random' 為雜訊)
        """
        super().__init__()
        self.eraser = T.RandomErasing(p=p, scale=scale, ratio=ratio, value=value, inplace=False)

    @torch.no_grad()
    def forward(self, batch_x):
        # batch_x shape: (B, 3, H, W)
        
        # 由於 RandomErasing 針對單張圖，我們用迴圈處理 (Batch size=4 很快)
        out = []
        for img in batch_x:
            out.append(self.eraser(img))
            
        return torch.stack(out)