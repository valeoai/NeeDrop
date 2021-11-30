
import torch
from torch import distributions as dist
from torch.autograd import Function

# Inherit from Function
class KLFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, target):
        ctx.save_for_backward(input, target)

        input = input.clamp(-10,10)

        exp_input = torch.exp(input)

        loss = torch.log(1+exp_input)

        mask = (target > 1e-7)
        loss[mask] = loss[mask] + (- target*input + target * torch.log(target))[mask]

        mask = (1-target > 1e-7)
        loss[mask] = loss[mask] + ((1-target) * torch.log(1-target))[mask]

        if torch.isnan(loss.sum()):
            print("is nan")
            print(input.min().item(), input.max().item())
            raise ValueError()
        if torch.isinf(loss.sum()):
            print("is inf")
            print(input.min().item(), input.max().item())
            raise ValueError()
        
        return loss

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):

        input, target = ctx.saved_tensors
        grad_input = grad_target = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.sigmoid(input) - target
            grad_input = grad_input * grad_output
        
        return grad_input, grad_target


# Inherit from Function
class JointKLFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input0, input1, target):
        ctx.save_for_backward(input0, input1, target)

        input0 = input0.clamp(-10,10)
        input1 = input1.clamp(-10,10)

        e0 = torch.exp(input0)
        e1 = torch.exp(input1)

        # loss = torch.log(1+e0) + torch.log(1+e1) - target*torch.log(1+e0*e1) - (1-target) * torch.log(e0+e1) + target * torch.log(target) + (1-target) * torch.log(1-target)
        loss = torch.log(1+e0) + torch.log(1+e1)

        mask = (target > 1e-7)
        loss[mask] = loss[mask] + (- target*torch.log(1+e0*e1) + target * torch.log(target))[mask]

        mask = (1-target > 1e-7)
        loss[mask] = loss[mask] + (-(1-target) * torch.log(e0+e1) + (1-target) * torch.log(1-target))[mask]

        if torch.isnan(loss.sum()):
            print("is nan")
            print(input0.min().item(), input1.min().item())
            print(input0.max().item(), input1.max().item())
            raise ValueError()
        if torch.isinf(loss.sum()):
            print("is inf")
            print(input0.min().item(), input1.min().item())
            print(input0.max().item(), input1.max().item())
            raise ValueError()
        
        return loss

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):

        input0, input1, target = ctx.saved_tensors
        grad_input0 = grad_input1 = grad_target = None

        if ctx.needs_input_grad[0]:
            grad_input0 = torch.sigmoid(input0)
            mask = (target > 1e-7)
            grad_input0[mask] = grad_input0[mask] - (target * torch.sigmoid(input0+input1))[mask]
            mask = (1-target > 1e-7)
            grad_input0[mask] = grad_input0[mask] - ((1-target) * torch.sigmoid(input0-input1))[mask]
            grad_input0 = grad_input0 * grad_output
        if ctx.needs_input_grad[1]:
            grad_input1 = torch.sigmoid(input1)
            mask = (target > 1e-7)
            grad_input1[mask] = grad_input1[mask] - (target * torch.sigmoid(input0+input1))[mask]
            mask = (1-target > 1e-7)
            grad_input1[mask] = grad_input1[mask] - ((1-target) * torch.sigmoid(input1-input0))[mask]
            grad_input1 = grad_input1 * grad_output
        
        return grad_input0, grad_input1, grad_target


kl_loss = KLFunction.apply
joint_kl_loss = JointKLFunction.apply


class UShareLoss(torch.nn.Module):

    def __init__(self, alpha_diff=1, alpha_same=1, alpha_box=1, alpha_code_regul=0):
        super().__init__()
        self.alpha_diff = alpha_diff
        self.alpha_same = alpha_same
        self.alpha_box = alpha_box
        self.alpha_code_regul = alpha_code_regul

    
    def forward(self, outputs_non_manifold_all, n_points, ids_pts2pts, return_all_losses_values=False):

        outputs_non_manifold_h0 = outputs_non_manifold_all[:,:n_points]
        outputs_non_manifold_h1 = outputs_non_manifold_all[:,n_points:2*n_points]
        outputs_non_manifold = outputs_non_manifold_all[:,2*n_points:]

        # loss for the points on each side of the shape
        loss_points_diff_sides = joint_kl_loss(outputs_non_manifold_h0, outputs_non_manifold_h1, torch.zeros_like(outputs_non_manifold_h0)).mean()

        # loss for points on the same side of the shape
        outputs_space_corresp = torch.gather(outputs_non_manifold_all, dim=1, index=ids_pts2pts)
        loss_points_same_sides = joint_kl_loss(outputs_non_manifold, outputs_space_corresp, torch.ones_like(outputs_non_manifold)).mean()

        # box loss
        outputs_box = outputs_non_manifold[:,-8:].unsqueeze(2).repeat(1,1,8).reshape(outputs_non_manifold.shape[0], -1)
        outputs_box_corresp = outputs_non_manifold[:,-8:].unsqueeze(1).repeat(1,8,1).reshape(outputs_non_manifold.shape[0], -1)
        loss_box = joint_kl_loss(outputs_box, outputs_box_corresp, torch.ones_like(outputs_box)).mean()

        loss =  self.alpha_diff * loss_points_diff_sides + \
                self.alpha_same * loss_points_same_sides + \
                self.alpha_box * loss_box

        if return_all_losses_values:

            ret_dict = {
                        "loss_diff": loss_points_diff_sides.detach().cpu().item(),
                        "loss_same": loss_points_same_sides.detach().cpu().item(),
                        "loss_box": loss_box.detach().cpu().item(),
                        "loss": loss.detach().cpu().item()
                        }

            return loss, ret_dict

        return loss
