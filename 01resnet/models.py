
from __future__ import print_function, division
import torch
import torch.nn as nn
import torchvision
# import daseg
# import daseg_bone


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads)
        self.layernorm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(rate)
        

    def forward(self, inputs):
        attn_output, attn_output_weights = self.att(inputs,inputs ,inputs)
        attn_output = self.dropout(attn_output)
        out = self.layernorm(inputs * attn_output)

        return out



class Age_assess(nn.Module):
    def __init__(self, num=12):
        super(Age_assess, self).__init__()

        self.model_inte = torchvision.models.resnet50(pretrained=True)        
        # self.model_inte.fc = nn.Linear(self.model_inte.fc.in_features , 18*15)
        for param in self.model_inte.parameters():
            param.requires_grad = True
        
        self.model_bone = torchvision.models.resnet50(pretrained=True)        
        # self.model_bone.fc = nn.Linear(self.model_bone.fc.in_features , 18*15)
        for param in self.model_bone.parameters():
            param.requires_grad = True
        
        self.model_sure = torchvision.models.resnet50(pretrained=True)        
        # self.model_sure.fc = nn.Linear(self.model_sure.fc.in_features , 18*15)
        for param in self.model_sure.parameters():
            param.requires_grad = True
        
    
        
        
        self.transformer_inte  = TransformerBlock(2048,8)
        self.transformer_bone  = TransformerBlock(2048,8)
        self.transformer_sure  = TransformerBlock(2048,8)
  
        
        # self.a1 = nn.Parameter(torch.ones(1))
        # self.a2 = nn.Parameter(torch.ones(1))
        # self.a3 = nn.Parameter(torch.ones(1))
        # self.a4 = nn.Parameter(torch.ones(1))
        # self.a5 = nn.Parameter(torch.ones(1))


        self.inte_evaluate = nn.Sequential(nn.Linear(2048,1024),nn.Linear(1024,512),nn.Linear(512,1))
        self.bone_evaluate = nn.Sequential(nn.Linear(2048,1024),nn.Linear(1024,512),nn.Linear(512,1))
        self.sure_evaluate = nn.Sequential(nn.Linear(2048,1024),nn.Linear(1024,512),nn.Linear(512,1))
        
        self.intea = nn.Parameter(torch.ones(1))
        self.bonea = nn.Parameter(torch.ones(1))
        self.surea = nn.Parameter(torch.ones(1))

        
        self.coarse_ranking = nn.Sequential(nn.Linear(2,20),nn.Linear(20,1))
        self.fine_evaluate = nn.Sequential(nn.Linear(810,100),nn.Linear(100,1))
        self.sex_info = nn.Sequential(nn.Linear(1,256),nn.Linear(256,1))
        # self.final_age =nn.Linear(33,1)

    def forward(self, sex, sa_ia, ba_ia, inte, bone, sure):
        # C1_ratio = self.a1*sa_ia
        # C2_ratio = self.a2*ba_ia
        # C_fea = torch.cat((sa_ia,ba_ia),dim=-1)
        # print(C_fea.shape)
        # coarse_age = self.coarse_ranking(C_fea) + ba_ia*10
        # print(coarse_age.shape)
        
        coarse_age = ba_ia
        # print(ba_ia.shape)
        

        #inte_fea = self.model_inte(inte.to(torch.float32))
        output = self.model_inte.conv1(inte.to(torch.float32))
        output = self.model_inte.bn1(output)
        output = self.model_inte.relu(output)
        output = self.model_inte.maxpool(output)
        output = self.model_inte.layer1(output)
        output = self.model_inte.layer2(output)
        output = self.model_inte.layer3(output)
        output = self.model_inte.layer4(output)
        inte_fea = self.model_inte.avgpool(output).squeeze(dim=-1).transpose(-2,-1)
        # print(inte_fea.shape)

  
        inte_att = self.transformer_inte(inte_fea)
        mean, std = torch.mean(inte_att), torch.std(inte_att)
        inte_att  = (inte_att-mean)/std  
        inte_fea = inte_fea* torch.clamp(inte_att,0.6,1)
        
        
        # bone_fea = self.model_bone(bone.to(torch.float32))
        output = self.model_bone.conv1(bone.to(torch.float32))
        output = self.model_bone.bn1(output)
        output = self.model_bone.relu(output)
        output = self.model_bone.maxpool(output)
        output = self.model_bone.layer1(output)
        output = self.model_bone.layer2(output)
        output = self.model_bone.layer3(output)
        output = self.model_bone.layer4(output)
        bone_fea = self.model_bone.avgpool(output).squeeze(dim=-1).transpose(-2,-1)
        bone_att = self.transformer_bone(bone_fea)
        mean, std = torch.mean(bone_att), torch.std(bone_att)
        bone_att  = (bone_att-mean)/std  
        bone_fea = bone_fea* torch.clamp(bone_att,0.8,1)
        
        # sure_fea = self.model_sure(sure.to(torch.float32))
        output = self.model_sure.conv1(sure.to(torch.float32))
        output = self.model_sure.bn1(output)
        output = self.model_sure.relu(output)
        output = self.model_sure.maxpool(output)
        output = self.model_sure.layer1(output)
        output = self.model_sure .layer2(output)
        output = self.model_sure.layer3(output)
        output = self.model_sure.layer4(output)
        sure_fea = self.model_sure.avgpool(output).squeeze(dim=-1).transpose(-2,-1)
        sure_att = self.transformer_sure(sure_fea)
        mean, std = torch.mean(sure_att), torch.std(sure_att)
        sure_att  = (sure_att-mean)/std  
        sure_fea = sure_fea* torch.clamp(sure_att,0.6,1)
        
        # print(sure_fea.shape)
        
        # inte_fea = inte_fea.reshape(inte_fea.shape[0],15,18)
        # bone_fea = bone_fea.reshape(inte_fea.shape[0],15,18)
        # sure_fea = sure_fea.reshape(inte_fea.shape[0],15,18)
        
        # G19_fea = torch.cat((inte_fea[:,0:9,:],bone_fea[:,0:9,:],sure_fea[:,0:9,:]),dim=1)
        # G1016_fea = torch.cat((inte_fea[:,9:14,:],bone_fea[:,9:14,:],sure_fea[:,9:14,:]),dim=1)
        # G1718_fea = torch.cat((inte_fea[:,14:15,:],bone_fea[:,14:15,:],sure_fea[:,14:15,:]),dim=1)
        
        
        # # G1_fea = self.a3*G19_fea
        # # G2_fea = self.a4*G1016_fea
        # # G3_fea = self.a5*G1718_fea
        
        
        # G1_fea = G19_fea
        # G2_fea = G1016_fea
        # G3_fea = G1718_fea
        
        
        # G_fea = torch.cat((G1_fea,G2_fea,G3_fea),dim=1)
        # G_fea = G_fea.reshape(G_fea.shape[0],-1)
        

        # fine_age = self.fine_evaluate(G_fea)
        # print(fine_age)
        
        
        fine_age = self.intea*self.inte_evaluate(inte_fea)+self.bonea*self.bone_evaluate(bone_fea) +self.surea*self.sure_evaluate(sure_fea)
        

        sex_age = self.sex_info(sex)

        # age = torch.cat((coarse_age,fine_age,sex_age),dim=1)   
        # out_age = self.final_age(age)
        

        return coarse_age, fine_age, sex_age