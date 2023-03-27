from numpy import dtype
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor = None, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha.to(torch.float32)
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets, multiclass=True):
        if self.alpha is None:
            self.alpha = torch.ones([inputs.shape[1]])
        if not multiclass:
            CE_loss = F.binary_cross_entropy(inputs, targets.float(), reduction='none')
        else:
            CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        weights = torch.ones(targets.shape, device=CE_loss.device)
        self.alpha = self.alpha.to(CE_loss.device)
        for i in range(self.alpha.shape[0]):
            weights[targets == i] = self.alpha[i]
        F_loss = weights * (1-pt)**self.gamma * CE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class TverskyLoss(nn.Module):

    def __init__(self, alpha=0.2, background_channel=0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # weight to background
        self.background_channel = background_channel
        self.smooth = 1e-6

    def tversky(self, y_true, y_pred):
        y_true_pos = torch.flatten(y_true)
        y_pred_pos = torch.flatten(y_pred)
        true_pos = torch.sum(y_true_pos * y_pred_pos)
        false_neg = torch.sum(y_true_pos * (1-y_pred_pos))
        false_pos = torch.sum((1-y_true_pos)*y_pred_pos)
        # return (true_pos + self.smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + self.smooth)
        return (true_pos + self.smooth)/(true_pos + false_neg + false_pos + self.smooth)

    def forward(self, y_pred, y_true):
        tversky_value = 0
        for channel in range(y_pred.shape[1]):
            if channel == self.background_channel:
                tversky_value += self.alpha * self.tversky(y_pred[:, channel, ...], y_true[:, channel, ...])
            else:
                tversky_value += (1-self.alpha) * self.tversky(y_pred[:, channel, ...], y_true[:, channel, ...])
        return 1 - tversky_value

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def dice_coeff(self, input: torch.Tensor, target: torch.Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
        # Average of Dice coefficient for all batches, or for a single mask
        assert input.size() == target.size()
        if input.dim() == 2 and reduce_batch_first:
            raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

        if input.dim() == 2 or reduce_batch_first:
            inter = torch.dot(input.reshape(-1), target.reshape(-1))
            sets_sum = torch.sum(input) + torch.sum(target)
            if sets_sum.item() == 0:
                sets_sum = 2 * inter

            return (2 * inter + epsilon) / (sets_sum + epsilon)
        else:
            # compute and average metric for each batch element
            dice = 0
            for i in range(input.shape[0]):
                dice += self.dice_coeff(input[i, ...], target[i, ...])
            return dice / input.shape[0]


    def multiclass_dice_coeff(self, input: torch.Tensor, target: torch.Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
        # Average of Dice coefficient for all classes
        assert input.size() == target.size()
        dice = 0
        for channel in range(input.shape[1]):
            dice += self.dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)
        return dice / input.shape[1]


    def forward(self, input: torch.Tensor, target: torch.Tensor, multiclass: bool = False):
        # Dice loss (objective to minimize) between 0 and 1
        assert input.size() == target.size()
        fn = self.multiclass_dice_coeff if multiclass else self.dice_coeff
        return 1 - fn(input, target, reduce_batch_first=True)

class OnlineBootstrappingLoss(nn.Module):
    def __init__(self, class_num, hard_num=512):
        super(OnlineBootstrappingLoss, self).__init__()
        self.keep_number = hard_num
        self.cross_entropy = nn.CrossEntropyLoss()
        self.bce = nn.BCELoss()
        self.count = torch.zeros([class_num])

    def single_class_ob_loss(self, inputs, targets):
        distance = torch.abs(inputs-targets.float())
        distance = distance.reshape([distance.shape[0], -1])
        distance_arg = distance.argsort(dim=1).cpu().numpy()
        reshaped_inputs = inputs.reshape([inputs.shape[0], -1])
        reshaped_targets = targets.reshape([inputs.shape[0], -1])
        inputs_mask = torch.zeros(reshaped_inputs.shape, dtype=torch.bool)
        targets_mask = torch.zeros(reshaped_targets.shape, dtype=torch.bool)
        for batch in range(inputs_mask.shape[0]):
            inputs_mask[batch, distance_arg[batch, -self.keep_number:]] = True
            targets_mask[batch, distance_arg[batch, -self.keep_number:]] = True
        
        loss = self.bce(reshaped_inputs[inputs_mask].reshape([inputs.shape[0], self.keep_number]), 
                        reshaped_targets[targets_mask].reshape([targets.shape[0], self.keep_number]))
        return loss

    def multi_class_ob_loss(self, inputs, targets):
        distance = torch.sum(torch.abs(inputs-F.one_hot(targets, inputs.shape[1]).permute(0, 3, 1, 2).float()), dim=1)
        distance = distance.reshape([distance.shape[0], -1])
        distance_arg = distance.argsort(dim=1).cpu().numpy()
        reshaped_inputs = inputs.reshape([inputs.shape[0], inputs.shape[1], -1])
        reshaped_targets = targets.reshape([inputs.shape[0], -1])
        inputs_mask = torch.zeros(reshaped_inputs.shape, dtype=torch.bool)
        targets_mask = torch.zeros(reshaped_targets.shape, dtype=torch.bool)
        for batch in range(inputs_mask.shape[0]):
            inputs_mask[batch, :, distance_arg[batch, -self.keep_number:]] = True
            targets_mask[batch, distance_arg[batch, -self.keep_number:]] = True

        compute_targets = reshaped_targets[targets_mask]
        
        for i in range(inputs.shape[1]):
            self.count[i] += torch.sum(compute_targets == i).cpu()
        
        loss = self.cross_entropy(reshaped_inputs[inputs_mask].reshape([*inputs.shape[:2], self.keep_number]), 
                                  reshaped_targets[targets_mask].reshape([targets.shape[0], self.keep_number]))
        return loss

    def forward(self, inputs, targets, multiclass=True):
        if multiclass:
            return self.multi_class_ob_loss(inputs, targets)
        else:
            return self.single_class_ob_loss(inputs, targets)
        

class MyLoss(nn.Module):
    def __init__(self, args, dataset, device, class_num):
        super(MyLoss, self).__init__()
        self.args = args
        self.device = device
        self.class_num = class_num
        self.softmax = nn.Softmax(dim=1)
        self.basic_loss_function =  nn.CrossEntropyLoss()
        self.dice_loss_function = DiceLoss()
        self.focal_loss_function = FocalLoss(dataset.weights)
        self.tversky_loss_function = TverskyLoss()
        self.online_bootstrap_loss_function = OnlineBootstrappingLoss(class_num=class_num, hard_num=args.hard_num)

    def forward(self, inputs, targets):
        # softmax
        inputs = self.softmax(inputs).float()

        # Basic Loss: CE
        if self.args.m_focal_loss | self.args.m_tversky_loss | self.args.m_dice_loss | self.args.m_ob_loss:
            basic_loss = torch.tensor(0, device=self.device)
        else:
            basic_loss = self.basic_loss_function(inputs, targets)
        
        # Focal Loss
        if self.args.m_focal_loss:
            focal_loss = self.focal_loss_function(inputs, targets) 
        else:
            focal_loss = torch.tensor(0, device=self.device)

        # Tversky Loss
        if self.args.m_tversky_loss:
            tversky_loss = self.tversky_loss_function(inputs,
                           F.one_hot(targets, self.class_num).permute(0, 3, 1, 2).float()) 
        else:
            tversky_loss = torch.tensor(0, device=self.device)

        # Dice Loss
        if self.args.m_dice_loss:
            dice_loss = self.dice_loss_function(inputs,
                           F.one_hot(targets, self.class_num).permute(0, 3, 1, 2).float(),
                           multiclass=True)
        else:
            dice_loss = torch.tensor(0, device=self.device)

        # # OnlineBootstrapp Loss
        if self.args.m_ob_loss:
            ob_loss = self.online_bootstrap_loss_function(inputs, targets)
        else:
            ob_loss = torch.tensor(0, device=self.device)

        loss = basic_loss + tversky_loss + focal_loss + dice_loss + ob_loss

        return loss, basic_loss, tversky_loss, focal_loss, dice_loss, ob_loss

class MyBinaryLoss(nn.Module):
    def __init__(self, args, dataset, device):
        super(MyBinaryLoss, self).__init__()
        self.args = args
        self.device = device
        self.sigmoid = nn.Sigmoid()
        self.basic_loss_function =  nn.CrossEntropyLoss()
        self.dice_loss_function = DiceLoss()
        self.focal_loss_function = FocalLoss(dataset.weights)
        self.tversky_loss_function = TverskyLoss()
        self.online_bootstrap_loss_function = OnlineBootstrappingLoss(class_num=1, hard_num=args.hard_num)

    def forward(self, inputs, targets):
        # softmax
        inputs = self.sigmoid(inputs).float()

        # Basic Loss: CE
        if self.args.b_focal_loss | self.args.b_tversky_loss | self.args.b_dice_loss | self.args.b_ob_loss:
            basic_loss = torch.tensor(0, device=self.device)
        else:
            basic_loss = self.basic_loss_function(inputs, targets)
        

        # Focal Loss
        if self.args.b_focal_loss:
            focal_loss = self.focal_loss_function(inputs, targets, False) 
        else:
            focal_loss = torch.tensor(0, device=self.device)
        
        # Dice Loss
        if self.args.b_dice_loss:
            dice_loss = self.dice_loss_function(inputs, targets.float(), multiclass=False)
        else:
            dice_loss = torch.tensor(0, device=self.device)

        # # OnlineBootstrapp Loss
        if self.args.b_ob_loss:
            ob_loss = self.online_bootstrap_loss_function(inputs, targets.float(), False)
        else:
            ob_loss = torch.tensor(0, device=self.device)

        loss = focal_loss + dice_loss + ob_loss + basic_loss

        return loss, focal_loss, dice_loss, ob_loss

if __name__ == "__main__":
    pred = torch.rand([4, 7, 256, 256])
    gt = torch.rand([4, 7, 256, 256])
    tversky_loss = TverskyLoss()
    loss = tversky_loss(pred, gt)
    print(loss)
