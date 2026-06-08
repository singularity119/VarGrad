import copy
import random
from abc import abstractmethod
from typing import Dict, List, Tuple, Union

import cvxpy as cp
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize, least_squares

from methods.min_norm_solvers import MinNormSolver, gradient_normalizers

EPS = 1e-8 # for numerical stability


class WeightMethod:
    def __init__(self, n_tasks: int, device: torch.device, max_norm = 1.0):
        super().__init__()
        self.n_tasks = n_tasks
        self.device = device
        self.max_norm = max_norm

    @abstractmethod
    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ],
        last_shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
        representation: Union[torch.nn.parameter.Parameter, torch.Tensor],
        **kwargs,
    ):
        pass

    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        last_shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        representation: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, None], Union[dict, None]]:
        """

        Parameters
        ----------
        losses :
        shared_parameters :
        task_specific_parameters :
        last_shared_parameters : parameters of last shared layer/block
        representation : shared representation
        kwargs :

        Returns
        -------
        Loss, extra outputs
        """
        loss, extra_outputs = self.get_weighted_loss(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            last_shared_parameters=last_shared_parameters,
            representation=representation,
            **kwargs,
        )

        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)

        loss.backward()
        return loss, extra_outputs

    def __call__(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ):
        return self.backward(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            **kwargs,
        )

    def parameters(self) -> List[torch.Tensor]:
        """return learnable parameters"""
        return []


class FAMO(WeightMethod):
    """Linear scalarization baseline L = sum_j w_j * l_j where l_j is the loss for task j and w_h"""

    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        gamma: float = 1e-5,
        w_lr: float = 0.025,
        task_weights: Union[List[float], torch.Tensor] = None,
        max_norm: float = 1.0,
    ):
        super().__init__(n_tasks, device=device)
        self.min_losses = torch.zeros(n_tasks).to(device)
        self.w = torch.tensor([0.0] * n_tasks, device=device, requires_grad=True)
        self.w_opt = torch.optim.Adam([self.w], lr=w_lr, weight_decay=gamma)
        self.max_norm = max_norm
    
    def set_min_losses(self, losses):
        self.min_losses = losses

    def get_weighted_loss(self, losses, **kwargs):
        self.prev_loss = losses
        z = F.softmax(self.w, -1)
        D = losses - self.min_losses + 1e-8
        c = (z / D).sum().detach()
        loss = (D.log() * z / c).sum()
        return loss, {"weights": z, "logits": self.w.detach().clone()}

    def update(self, curr_loss):
        delta = (self.prev_loss - self.min_losses + 1e-8).log() - \
                (curr_loss      - self.min_losses + 1e-8).log()
        with torch.enable_grad():
            d = torch.autograd.grad(F.softmax(self.w, -1),
                                    self.w,
                                    grad_outputs=delta.detach())[0]
        self.w_opt.zero_grad()
        self.w.grad = d
        self.w_opt.step()
    
    def get_imbalance(self, losses):
        # acquire weights sum=1
        z = F.softmax(self.w, -1)
        return z.max() / z.min()
    
    def update_prev_loss(self, losses):
        self.prev_loss = losses


class NashMTL(WeightMethod):
    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        max_norm: float = 1.0,
        update_weights_every: int = 1,
        optim_niter=20,
    ):
        super(NashMTL, self).__init__(
            n_tasks=n_tasks,
            device=device,
        )

        self.optim_niter = optim_niter
        self.update_weights_every = update_weights_every
        self.max_norm = max_norm

        self.prvs_alpha_param = None
        self.normalization_factor = np.ones((1,))
        self.init_gtg = self.init_gtg = np.eye(self.n_tasks)
        self.step = 0.0
        self.prvs_alpha = np.ones(self.n_tasks, dtype=np.float32)
        
        # 添加 Vargrad 相关参数
        self.last_grads = None
        self.exp_avg = None


    def _stop_criteria(self, gtg, alpha_t):
        return (
            (self.alpha_param.value is None)
            or (np.linalg.norm(gtg @ alpha_t - 1 / (alpha_t + 1e-10)) < 1e-3)
            or (
                np.linalg.norm(self.alpha_param.value - self.prvs_alpha_param.value)
                < 1e-6
            )
        )

    def solve_optimization(self, gtg: np.array):
        self.G_param.value = gtg
        self.normalization_factor_param.value = self.normalization_factor

        alpha_t = self.prvs_alpha
        for _ in range(self.optim_niter):
            self.alpha_param.value = alpha_t
            self.prvs_alpha_param.value = alpha_t

            try:
                self.prob.solve(solver=cp.ECOS, warm_start=True, max_iters=100)
            except:
                self.alpha_param.value = self.prvs_alpha_param.value

            if self._stop_criteria(gtg, alpha_t):
                break

            alpha_t = self.alpha_param.value

        if alpha_t is not None:
            self.prvs_alpha = alpha_t

        return self.prvs_alpha

    def _calc_phi_alpha_linearization(self):
        G_prvs_alpha = self.G_param @ self.prvs_alpha_param
        prvs_phi_tag = 1 / self.prvs_alpha_param + (1 / G_prvs_alpha) @ self.G_param
        phi_alpha = prvs_phi_tag @ (self.alpha_param - self.prvs_alpha_param)
        return phi_alpha

    def _init_optim_problem(self):
        self.alpha_param = cp.Variable(shape=(self.n_tasks,), nonneg=True)
        self.prvs_alpha_param = cp.Parameter(
            shape=(self.n_tasks,), value=self.prvs_alpha
        )
        self.G_param = cp.Parameter(
            shape=(self.n_tasks, self.n_tasks), value=self.init_gtg
        )
        self.normalization_factor_param = cp.Parameter(
            shape=(1,), value=np.array([1.0])
        )

        self.phi_alpha = self._calc_phi_alpha_linearization()

        G_alpha = self.G_param @ self.alpha_param
        constraint = []
        for i in range(self.n_tasks):
            constraint.append(
                -cp.log(self.alpha_param[i] * self.normalization_factor_param)
                - cp.log(G_alpha[i])
                <= 0
            )
        obj = cp.Minimize(
            cp.sum(G_alpha) + self.phi_alpha / self.normalization_factor_param
        )
        self.prob = cp.Problem(obj, constraint)

    def get_weighted_loss(
        self,
        losses,
        shared_parameters,
        beta=None,
        **kwargs,
    ):
        extra_outputs = dict()
        if self.step == 0:
            self._init_optim_problem()

        if (self.step % self.update_weights_every) == 0:
            self.step += 1

            # 第一步：获取原始梯度
            grads = {}
            for i, loss in enumerate(losses):
                g = list(
                    torch.autograd.grad(
                        loss,
                        shared_parameters,
                        retain_graph=True,
                    )
                )
                grad = torch.cat([torch.flatten(grad) for grad in g])
                grads[i] = grad

            # 第二步：将梯度转换为矩阵形式，用于 MARS 处理
            grad_dims = []
            for param in shared_parameters:
                grad_dims.append(param.data.numel())
            grads_matrix = torch.Tensor(sum(grad_dims), self.n_tasks).to(self.device)
            
            # 填充梯度矩阵
            for i in range(self.n_tasks):
                grads_matrix[:, i].copy_(grads[i])

            # 第三步：用 Vargrad 处理梯度
            grads_matrix, self.last_grads, self.exp_avg = Vargrad(
                grads_matrix,
                last_grads=self.last_grads,
                exp_avg=self.exp_avg,
                step=self.step,
                beta=beta,
                gamma=1.0,
                eps=1e-8
            )
            
            # 第四步：将处理后的梯度转换回字典形式
            processed_grads = {}
            for i in range(self.n_tasks):
                processed_grads[i] = grads_matrix[:, i]

            # 第五步：构建 GTG 矩阵（用处理后的梯度）
            G = torch.stack(tuple(v for v in processed_grads.values()))
            GTG = torch.mm(G, G.t())

            self.normalization_factor = (
                torch.norm(GTG).detach().cpu().numpy().reshape((1,))
            )
            GTG = GTG / self.normalization_factor.item()
            alpha = self.solve_optimization(GTG.cpu().detach().numpy())
            alpha = torch.from_numpy(alpha)

        else:
            self.step += 1
            alpha = self.prvs_alpha
            # 修复原始代码的 bug：在 else 分支中也需要定义 GTG
            # GTG = torch.eye(self.n_tasks, device=self.device)

        weighted_loss = sum([losses[i] * alpha[i] for i in range(len(alpha))])
        extra_outputs["weights"] = alpha
        extra_outputs["GTG"] = GTG.detach().cpu().numpy()
        return weighted_loss, extra_outputs


class LinearScalarization(WeightMethod):
    """Linear scalarization baseline L = sum_j w_j * l_j where l_j is the loss for task j and w_h"""

    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        task_weights: Union[List[float], torch.Tensor] = None,
    ):
        super().__init__(n_tasks, device=device)
        if task_weights is None:
            task_weights = torch.ones((n_tasks,))
        if not isinstance(task_weights, torch.Tensor):
            task_weights = torch.tensor(task_weights)
        assert len(task_weights) == n_tasks
        self.task_weights = task_weights.to(device)

    def get_weighted_loss(self, losses, **kwargs):
        loss = torch.sum(losses * self.task_weights)
        return loss, dict(weights=self.task_weights)


class ScaleInvariantLinearScalarization(WeightMethod):
    """Linear scalarization baseline L = sum_j w_j * l_j where l_j is the loss for task j and w_h"""

    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        task_weights: Union[List[float], torch.Tensor] = None,
    ):
        super().__init__(n_tasks, device=device)
        if task_weights is None:
            task_weights = torch.ones((n_tasks,))
        if not isinstance(task_weights, torch.Tensor):
            task_weights = torch.tensor(task_weights)
        assert len(task_weights) == n_tasks
        self.task_weights = task_weights.to(device)

    def get_weighted_loss(self, losses, **kwargs):
        loss = torch.sum(torch.log(losses) * self.task_weights)
        return loss, dict(weights=self.task_weights)


class MGDA(WeightMethod):
    """Based on the official implementation of: Multi-Task Learning as Multi-Objective Optimization
    Ozan Sener, Vladlen Koltun
    Neural Information Processing Systems (NeurIPS) 2018
    https://github.com/intel-isl/MultiObjectiveOptimization

    """

    def __init__(
        self, n_tasks, device: torch.device, params="shared", normalization="none",
        max_norm=1.0,
    ):
        super().__init__(n_tasks, device=device)
        self.solver = MinNormSolver()
        assert params in ["shared", "last", "rep"]
        self.params = params
        assert normalization in ["norm", "loss", "loss+", "none"]
        self.normalization = normalization
        self.max_norm = max_norm
        
        # 添加 Vargrad 相关参数
        self.last_grads = None
        self.exp_avg = None
        self.step = 1

    @staticmethod
    def _flattening(grad):
        return torch.cat(
            tuple(
                g.reshape(
                    -1,
                )
                for i, g in enumerate(grad)
            ),
            dim=0,
        )

    def get_weighted_loss(
        self,
        losses,
        shared_parameters=None,
        last_shared_parameters=None,
        representation=None,
        beta=None,
        **kwargs,
    ):
        # 第一步：获取原始梯度（保持 MGDA 原有逻辑）
        grads = {}
        params = dict(
            rep=representation, shared=shared_parameters, last=last_shared_parameters
        )[self.params]
        
        for i, loss in enumerate(losses):
            g = list(
                torch.autograd.grad(
                    loss,
                    params,
                    retain_graph=True,
                )
            )
            grads[i] = [torch.flatten(grad) for grad in g]

        # 第二步：应用梯度归一化（保持 MGDA 原有逻辑）
        gn = gradient_normalizers(grads, losses, self.normalization)
        for t in range(self.n_tasks):
            for gr_i in range(len(grads[t])):
                grads[t][gr_i] = grads[t][gr_i] / gn[t]

        # 第三步：将梯度转换为矩阵形式，用于 MARS 处理
        grad_dims = []
        for param in params:
            grad_dims.append(param.data.numel())
        grads_matrix = torch.Tensor(sum(grad_dims), self.n_tasks).to(self.device)
        
        # 填充梯度矩阵
        for i in range(self.n_tasks):
            cnt = 0
            for gr_i in range(len(grads[i])):
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[: cnt + 1])
                grads_matrix[beg:en, i].copy_(grads[i][gr_i].data.view(-1))
                cnt += 1

        # 第四步：用 Vargrad 处理梯度
        grads_matrix, self.last_grads, self.exp_avg = Vargrad(
            grads_matrix,
            last_grads=self.last_grads,
            exp_avg=self.exp_avg,
            step=self.step,
            beta=beta,
            gamma=1.0,
            eps=1e-8
        )
        
        # 第五步：将处理后的梯度转换回列表形式
        processed_grads = {}
        for i in range(self.n_tasks):
            processed_grads[i] = []
            cnt = 0
            for gr_i in range(len(grads[i])):
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[: cnt + 1])
                processed_grads[i].append(grads_matrix[beg:en, i].view_as(grads[i][gr_i]))
                cnt += 1

        # 第六步：用 MGDA 求解器（保持原有逻辑）
        sol, min_norm = self.solver.find_min_norm_element(
            [processed_grads[t] for t in range(len(processed_grads))]
        )
        sol = sol * self.n_tasks  # make sure it sums to self.n_tasks
        weighted_loss = sum([losses[i] * sol[i] for i in range(len(sol))])

        return weighted_loss, dict(weights=torch.from_numpy(sol.astype(np.float32)))


class LOG_MGDA(WeightMethod):
    """Based on the official implementation of: Multi-Task Learning as Multi-Objective Optimization
    Ozan Sener, Vladlen Koltun
    Neural Information Processing Systems (NeurIPS) 2018
    https://github.com/intel-isl/MultiObjectiveOptimization

    """

    def __init__(self, n_tasks, device: torch.device, params="shared", normalization="none",
        max_norm=1.0,
    ):
        super().__init__(n_tasks, device=device)
        self.solver = MinNormSolver()
        assert params in ["shared", "last", "rep"]
        self.params = params
        assert normalization in ["norm", "loss", "loss+", "none"]
        self.normalization = normalization
        self.max_norm = max_norm

    @staticmethod
    def _flattening(grad):
        return torch.cat(
            tuple(
                g.reshape(
                    -1,
                )
                for i, g in enumerate(grad)
            ),
            dim=0,
        )

    def get_weighted_loss(
        self,
        losses,
        shared_parameters=None,
        last_shared_parameters=None,
        representation=None,
        **kwargs,
    ):
        """

        Parameters
        ----------
        losses :
        shared_parameters :
        last_shared_parameters :
        representation :
        kwargs :

        Returns
        -------

        """
        # Our code
        grads = {}
        params = dict(
            rep=representation, shared=shared_parameters, last=last_shared_parameters
        )[self.params]
        for i, loss in enumerate(losses):
            g = list(
                torch.autograd.grad(
                    (loss + 1e-8).log(),
                    params,
                    retain_graph=True,
                )
            )
            # Normalize all gradients, this is optional and not included in the paper.

            grads[i] = [torch.flatten(grad) for grad in g]

        gn = gradient_normalizers(grads, losses, self.normalization)
        for t in range(self.n_tasks):
            for gr_i in range(len(grads[t])):
                grads[t][gr_i] = grads[t][gr_i] / gn[t]

        sol, min_norm = self.solver.find_min_norm_element(
            [grads[t] for t in range(len(grads))]
        )
        #sol = sol * self.n_tasks  # make sure it sums to self.n_tasks
        c = sum([ sol[i] / (losses[i] + 1e-8).detach() for i in range(len(sol))])
        weighted_loss = sum([(losses[i] + 1e-8).log() * sol[i] / c for i in range(len(sol))])
        return weighted_loss, dict(weights=torch.from_numpy(sol.astype(np.float32)))


class STL(WeightMethod):
    """Single task learning"""

    def __init__(self, n_tasks, device: torch.device, main_task):
        super().__init__(n_tasks, device=device)
        self.main_task = main_task
        self.weights = torch.zeros(n_tasks, device=device)
        self.weights[main_task] = 1.0

    def get_weighted_loss(self, losses: torch.Tensor, **kwargs):
        assert len(losses) == self.n_tasks
        loss = losses[self.main_task]

        return loss, dict(weights=self.weights)


class Uncertainty(WeightMethod):
    """Implementation of `Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics`
    Source: https://github.com/yaringal/multi-task-learning-example/blob/master/multi-task-learning-example-pytorch.ipynb
    """

    def __init__(self, n_tasks, device: torch.device):
        super().__init__(n_tasks, device=device)
        self.logsigma = torch.tensor([0.0] * n_tasks, device=device, requires_grad=True)

    def get_weighted_loss(self, losses: torch.Tensor, **kwargs):
        loss = sum(
            [
                0.5 * (torch.exp(-logs) * loss + logs)
                for loss, logs in zip(losses, self.logsigma)
            ]
        )

        return loss, dict(
            weights=torch.exp(-self.logsigma)
        )  # NOTE: not exactly task weights

    def parameters(self) -> List[torch.Tensor]:
        return [self.logsigma]


class PCGrad(WeightMethod):
    """Modification of: https://github.com/WeiChengTseng/Pytorch-PCGrad/blob/master/pcgrad.py

    @misc{Pytorch-PCGrad,
      author = {Wei-Cheng Tseng},
      title = {WeiChengTseng/Pytorch-PCGrad},
      url = {https://github.com/WeiChengTseng/Pytorch-PCGrad.git},
      year = {2020}
    }

    """

    def __init__(self, n_tasks: int, device: torch.device, reduction="sum"):
        super().__init__(n_tasks, device=device)
        assert reduction in ["mean", "sum"]
        self.reduction = reduction

    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ):
        raise NotImplementedError

    def _set_pc_grads(self, losses, shared_parameters, task_specific_parameters=None):
        # shared part
        shared_grads = []
        for l in losses:
            shared_grads.append(
                torch.autograd.grad(l, shared_parameters, retain_graph=True)
            )

        if isinstance(shared_parameters, torch.Tensor):
            shared_parameters = [shared_parameters]
        non_conflict_shared_grads = self._project_conflicting(shared_grads)
        for p, g in zip(shared_parameters, non_conflict_shared_grads):
            p.grad = g

        # task specific part
        if task_specific_parameters is not None:
            task_specific_grads = torch.autograd.grad(
                losses.sum(), task_specific_parameters
            )
            if isinstance(task_specific_parameters, torch.Tensor):
                task_specific_parameters = [task_specific_parameters]
            for p, g in zip(task_specific_parameters, task_specific_grads):
                p.grad = g

    def _project_conflicting(self, grads: List[Tuple[torch.Tensor]]):
        pc_grad = copy.deepcopy(grads)
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = sum(
                    [
                        torch.dot(torch.flatten(grad_i), torch.flatten(grad_j))
                        for grad_i, grad_j in zip(g_i, g_j)
                    ]
                )
                if g_i_g_j < 0:
                    g_j_norm_square = (
                        torch.norm(torch.cat([torch.flatten(g) for g in g_j])) ** 2
                    )
                    for grad_i, grad_j in zip(g_i, g_j):
                        grad_i -= g_i_g_j * grad_j / g_j_norm_square

        merged_grad = [sum(g) for g in zip(*pc_grad)]
        if self.reduction == "mean":
            merged_grad = [g / self.n_tasks for g in merged_grad]

        return merged_grad

    def backward(
        self,
        losses: torch.Tensor,
        parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ):
        self._set_pc_grads(losses, shared_parameters, task_specific_parameters)
        # make sure the solution for shared params has norm <= self.eps
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        return None, {}  # NOTE: to align with all other weight methods


class CAGrad(WeightMethod):
    def __init__(self, n_tasks, device: torch.device, c=0.4, max_norm=1.0):
        super().__init__(n_tasks, device=device)
        self.c = c
        self.max_norm = max_norm

    def get_weighted_loss(
        self,
        losses,
        shared_parameters,
        **kwargs,
    ):
        """
        Parameters
        ----------
        losses :
        shared_parameters : shared parameters
        kwargs :
        Returns
        -------
        """
        # NOTE: we allow only shared params for now. Need to see paper for other options.
        grad_dims = []
        for param in shared_parameters:
            grad_dims.append(param.data.numel())
        grads = torch.Tensor(sum(grad_dims), self.n_tasks).to(self.device)

        for i in range(self.n_tasks):
            if i < self.n_tasks:
                losses[i].backward(retain_graph=True)
            else:
                losses[i].backward()
            self.grad2vec(shared_parameters, grads, grad_dims, i)
            # multi_task_model.zero_grad_shared_modules()
            for p in shared_parameters:
                p.grad = None

        g, GTG, w_cpu = self.cagrad(grads, alpha=self.c, rescale=1)
        self.overwrite_grad(shared_parameters, g, grad_dims)
        return GTG, w_cpu

    def cagrad(self, grads, alpha=0.5, rescale=1):
        GG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]
        g0_norm = (GG.mean() + 1e-8).sqrt()  # norm of the average gradient

        x_start = np.ones(self.n_tasks) / self.n_tasks
        bnds = tuple((0, 1) for x in x_start)
        cons = {"type": "eq", "fun": lambda x: 1 - sum(x)}
        A = GG.numpy()
        b = x_start.copy()
        c = (alpha * g0_norm + 1e-8).item()

        def objfn(x):
            return (
                x.reshape(1, self.n_tasks).dot(A).dot(b.reshape(self.n_tasks, 1))
                + c
                * np.sqrt(
                    x.reshape(1, self.n_tasks).dot(A).dot(x.reshape(self.n_tasks, 1))
                    + 1e-8
                )
            ).sum()

        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(grads.device)
        gw = (grads * ww.view(1, -1)).sum(1)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm + 1e-8)
        g = grads.mean(1) + lmbda * gw
        if rescale == 0:
            return g, GG.numpy(), w_cpu
        elif rescale == 1:
            return g / (1 + alpha ** 2), GG.numpy(), w_cpu
        else:
            return g / (1 + alpha), GG.numpy(), w_cpu

    @staticmethod
    def grad2vec(shared_params, grads, grad_dims, task):
        # store the gradients
        grads[:, task].fill_(0.0)
        cnt = 0
        # for mm in m.shared_modules():
        #     for p in mm.parameters():

        for param in shared_params:
            grad = param.grad
            if grad is not None:
                grad_cur = grad.data.detach().clone()
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[: cnt + 1])
                grads[beg:en, task].copy_(grad_cur.data.view(-1))
            cnt += 1

    def overwrite_grad(self, shared_parameters, newgrad, grad_dims):
        newgrad = newgrad * self.n_tasks  # to match the sum loss
        cnt = 0

        # for mm in m.shared_modules():
        #     for param in mm.parameters():
        for param in shared_parameters:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[: cnt + 1])
            this_grad = newgrad[beg:en].contiguous().view(param.data.size())
            param.grad = this_grad.data.clone()
            cnt += 1

    def backward(
        self,
        losses: torch.Tensor,
        parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ):
        GTG, w = self.get_weighted_loss(losses, shared_parameters)
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        return None, {"GTG": GTG, "weights": w}  # NOTE: to align with all other weight methods


class FairGrad(WeightMethod):
    def __init__(
        self,
        n_tasks,
        device: torch.device,
        alpha=1.0,
        max_norm=1.0,
        preprocessing="vargrad",
        scheduler="every_step",
        beta=0.85,
        psmgd_R=10,
        psmgd_alpha=0.5,
        psmgd_dynamic_threshold=1.0164316892623901,
        psmgd_dynamic_metric="refresh_rel_fro",
        psmgd_dynamic_direction="below",
    ):
        super().__init__(n_tasks, device=device)
        if preprocessing not in ["identity", "vargrad"]:
            raise ValueError(f"unknown preprocessing {preprocessing}.")
        if scheduler not in ["every_step", "psmgd_periodic", "psmgd_dynamic"]:
            raise ValueError(f"unknown scheduler {scheduler}.")
        if psmgd_dynamic_metric not in ["refresh_rel_fro", "step_rel_fro"]:
            raise ValueError(f"unknown psmgd_dynamic_metric {psmgd_dynamic_metric}.")
        if psmgd_dynamic_direction not in ["above", "below"]:
            raise ValueError(
                f"unknown psmgd_dynamic_direction {psmgd_dynamic_direction}."
            )

        self.alpha = alpha
        self.max_norm = max_norm
        self.preprocessing = preprocessing
        self.scheduler = scheduler
        self.beta = beta
        self.gamma = 1.0
        self.psmgd_R = psmgd_R
        self.psmgd_alpha = psmgd_alpha
        self.psmgd_dynamic_threshold = psmgd_dynamic_threshold
        self.psmgd_dynamic_metric = psmgd_dynamic_metric
        self.psmgd_dynamic_direction = psmgd_dynamic_direction
        self.last_grads = None
        self.exp_avg = None
        self.step = 1
        self.weights = torch.ones(n_tasks, device=device, dtype=torch.float32)
        self.last_candidate_weights = self.weights.clone()
        self.prev_solver_task_grads = None
        self.last_refresh_task_grads = None
        self.last_refresh_step = -1
        self._latest_shared_grad = None
        self.grad_stats_history = []
        self.mars_stats_history = []

    @staticmethod
    def _to_parameter_list(parameters):
        if parameters is None:
            return []
        if isinstance(parameters, torch.Tensor):
            return [parameters]
        return list(parameters)

    @staticmethod
    def _flatten_grad_tuple(grads, parameters):
        flat_grads = []
        for grad, parameter in zip(grads, parameters):
            if grad is None:
                flat_grads.append(torch.zeros_like(parameter).reshape(-1))
            else:
                flat_grads.append(grad.reshape(-1))
        if not flat_grads:
            if parameters:
                return torch.zeros(0, device=parameters[0].device)
            return torch.zeros(0)
        return torch.cat(flat_grads, dim=0)

    def _collect_shared_task_grads(self, losses, shared_parameters):
        task_grads = []
        for loss in losses:
            grad_tuple = torch.autograd.grad(
                loss,
                shared_parameters,
                retain_graph=True,
                allow_unused=True,
            )
            task_grads.append(self._flatten_grad_tuple(grad_tuple, shared_parameters))
        return torch.stack(task_grads, dim=1)

    def _apply_preprocessing(self, raw_task_grads):
        if self.preprocessing != "vargrad":
            return raw_task_grads
        solver_task_grads, self.last_grads, self.exp_avg = Vargrad(
            raw_task_grads,
            last_grads=self.last_grads,
            exp_avg=self.exp_avg,
            step=self.step,
            beta=self.beta,
            gamma=self.gamma,
            eps=EPS,
        )
        return solver_task_grads

    def _scheduler_step(self):
        return self.step - 1

    @staticmethod
    def _float(value):
        return float(value.detach().cpu().item())

    @staticmethod
    def _float_list(value):
        if isinstance(value, np.ndarray):
            return [float(item) for item in value.tolist()]
        return [float(item) for item in value.detach().cpu().tolist()]

    def _compute_step_rel_fro(self, solver_task_grads):
        if self.prev_solver_task_grads is None:
            return 0.0
        current = solver_task_grads.detach()
        step_delta = current - self.prev_solver_task_grads
        step_diff = torch.norm(step_delta)
        prev_norm = torch.norm(self.prev_solver_task_grads)
        return self._float(step_diff / (prev_norm + EPS))

    def _compute_refresh_rel_fro(self, solver_task_grads):
        if self.last_refresh_task_grads is None:
            return 0.0
        current = solver_task_grads.detach()
        refresh_delta = current - self.last_refresh_task_grads
        refresh_diff = torch.norm(refresh_delta)
        last_refresh_u_norm = torch.norm(self.last_refresh_task_grads)
        return self._float(refresh_diff / (last_refresh_u_norm + EPS))

    def _compute_dynamic_refresh_score(self, solver_task_grads):
        if self.psmgd_dynamic_metric == "refresh_rel_fro":
            return self._compute_refresh_rel_fro(solver_task_grads)
        if self.psmgd_dynamic_metric == "step_rel_fro":
            return self._compute_step_rel_fro(solver_task_grads)
        raise ValueError(f"unknown psmgd_dynamic_metric {self.psmgd_dynamic_metric}.")

    def _should_call_solver(self, solver_task_grads, scheduler_step):
        if self.scheduler == "every_step":
            return True, False, 0.0
        if self.scheduler == "psmgd_periodic":
            return (scheduler_step % self.psmgd_R) == 0, False, 0.0
        if self.last_refresh_task_grads is None:
            return True, True, 0.0

        dynamic_refresh_score = self._compute_dynamic_refresh_score(solver_task_grads)
        if self.psmgd_dynamic_direction == "above":
            dynamic_refresh_triggered = (
                dynamic_refresh_score > self.psmgd_dynamic_threshold
            )
        else:
            dynamic_refresh_triggered = (
                dynamic_refresh_score <= self.psmgd_dynamic_threshold
            )
        return dynamic_refresh_triggered, dynamic_refresh_triggered, dynamic_refresh_score

    def _apply_scheduler(self, candidate_weights):
        candidate_weights = candidate_weights.to(self.device).float()
        self.last_candidate_weights = candidate_weights.detach().clone()

        if self.scheduler == "every_step":
            self.weights = candidate_weights
            return self.weights, True

        if self.last_refresh_step < 0:
            self.weights = candidate_weights
        else:
            self.weights = (
                self.psmgd_alpha * self.weights
                + (1.0 - self.psmgd_alpha) * candidate_weights
            )
        return self.weights, True

    def _build_u_telemetry(
        self,
        solver_task_grads,
        task_weights,
        candidate_weights,
        scheduler_step,
        updated_weights,
        solver_called,
        dynamic_refresh_triggered,
        dynamic_refresh_score,
    ):
        current = solver_task_grads.detach()
        u_norm = torch.norm(current)
        task_u_norms = current.norm(dim=0)

        if self.prev_solver_task_grads is None:
            step_diff = current.new_tensor(0.0)
            prev_norm = current.new_tensor(1.0)
            task_step_rel = torch.zeros_like(task_u_norms)
        else:
            step_delta = current - self.prev_solver_task_grads
            step_diff = torch.norm(step_delta)
            prev_norm = torch.norm(self.prev_solver_task_grads)
            task_step_diff = step_delta.norm(dim=0)
            prev_task_norms = self.prev_solver_task_grads.norm(dim=0)
            task_step_rel = task_step_diff / (prev_task_norms + EPS)

        step_rel = step_diff / (prev_norm + EPS)

        if self.last_refresh_task_grads is None:
            last_refresh_u_norm = current.new_tensor(0.0)
            last_refresh_task_u_norms = torch.zeros_like(task_u_norms)
            refresh_diff = current.new_tensor(0.0)
            refresh_rel = current.new_tensor(0.0)
        else:
            refresh_delta = current - self.last_refresh_task_grads
            refresh_diff = torch.norm(refresh_delta)
            last_refresh_u_norm = torch.norm(self.last_refresh_task_grads)
            last_refresh_task_u_norms = self.last_refresh_task_grads.norm(dim=0)
            refresh_rel = refresh_diff / (last_refresh_u_norm + EPS)

        task_u_norms_list = self._float_list(task_u_norms)
        task_step_rel_list = self._float_list(task_step_rel)
        telemetry = {
            "scheduler_step": int(scheduler_step),
            "u_norm_fro": self._float(u_norm),
            "step_diff_fro": self._float(step_diff),
            "step_rel_fro": self._float(step_rel),
            "task_u_norms": task_u_norms_list,
            "task_step_rel": task_step_rel_list,
            "step_max_task_rel": max(task_step_rel_list) if task_step_rel_list else 0.0,
            "step_sum_task_rel": float(sum(task_step_rel_list)),
            "last_refresh_step": int(self.last_refresh_step),
            "last_refresh_u_norm_fro": self._float(last_refresh_u_norm),
            "last_refresh_task_u_norms": self._float_list(
                last_refresh_task_u_norms
            ),
            "refresh_diff_fro": self._float(refresh_diff),
            "refresh_rel_fro": self._float(refresh_rel),
            "solver_called": bool(solver_called),
            "updated_weights": bool(updated_weights),
            "dynamic_refresh_metric": self.psmgd_dynamic_metric,
            "dynamic_refresh_direction": self.psmgd_dynamic_direction,
            "dynamic_refresh_score": float(dynamic_refresh_score),
            "dynamic_refresh_threshold": float(self.psmgd_dynamic_threshold),
            "dynamic_refresh_triggered": bool(dynamic_refresh_triggered),
            "weights": self._float_list(task_weights),
            "candidate_weights": self._float_list(candidate_weights),
        }

        self.prev_solver_task_grads = current.clone()
        if updated_weights:
            self.last_refresh_task_grads = current.clone()
            self.last_refresh_step = scheduler_step

        return telemetry

    def get_weighted_loss(
        self,
        losses,
        shared_parameters,
        **kwargs,
    ):
        shared_parameters = self._to_parameter_list(shared_parameters)
        extra_outputs = dict()
        if not shared_parameters:
            task_weights = self.weights.detach()
            weighted_loss = torch.sum(losses * task_weights)
            extra_outputs["weights"] = task_weights.detach().clone()
            extra_outputs["candidate_weights"] = self.last_candidate_weights.detach().clone()
            extra_outputs["updated_weights"] = False
            extra_outputs["solver_called"] = False
            extra_outputs["scheduler_step"] = self._scheduler_step()
            return weighted_loss, extra_outputs

        raw_task_grads = self._collect_shared_task_grads(losses, shared_parameters)
        solver_task_grads = self._apply_preprocessing(raw_task_grads)
        scheduler_step = self._scheduler_step()
        (
            solver_called,
            dynamic_refresh_triggered,
            dynamic_refresh_score,
        ) = self._should_call_solver(solver_task_grads, scheduler_step)

        if solver_called:
            _, GTG, w_cpu = self.fairgrad(solver_task_grads, alpha=self.alpha)
            candidate_weights = torch.from_numpy(w_cpu.astype(np.float32)).to(self.device)
            task_weights, updated_weights = self._apply_scheduler(candidate_weights)
        else:
            GTG = solver_task_grads.t().mm(solver_task_grads).detach().cpu().numpy()
            candidate_weights = self.last_candidate_weights
            task_weights = self.weights
            updated_weights = False

        self._latest_shared_grad = (
            solver_task_grads * task_weights.detach().view(1, -1)
        ).sum(dim=1) * self.n_tasks

        u_telemetry = self._build_u_telemetry(
            solver_task_grads=solver_task_grads,
            task_weights=task_weights,
            candidate_weights=self.last_candidate_weights,
            scheduler_step=scheduler_step,
            updated_weights=updated_weights,
            solver_called=solver_called,
            dynamic_refresh_triggered=dynamic_refresh_triggered,
            dynamic_refresh_score=dynamic_refresh_score,
        )
        self.step += 1

        weighted_loss = torch.sum(losses * task_weights.detach())
        extra_outputs["GTG"] = GTG
        extra_outputs["weights"] = task_weights.detach().clone()
        extra_outputs["candidate_weights"] = self.last_candidate_weights.detach().clone()
        extra_outputs["updated_weights"] = updated_weights
        extra_outputs["solver_called"] = solver_called
        extra_outputs["scheduler_step"] = scheduler_step
        extra_outputs["preprocessing"] = self.preprocessing
        extra_outputs["solver"] = "fairgrad"
        extra_outputs["scheduler"] = self.scheduler
        extra_outputs["dynamic_refresh_metric"] = self.psmgd_dynamic_metric
        extra_outputs["dynamic_refresh_direction"] = self.psmgd_dynamic_direction
        extra_outputs["dynamic_refresh_score"] = float(dynamic_refresh_score)
        extra_outputs["dynamic_refresh_threshold"] = float(
            self.psmgd_dynamic_threshold
        )
        extra_outputs["dynamic_refresh_triggered"] = dynamic_refresh_triggered
        extra_outputs["u_telemetry"] = u_telemetry
        extra_outputs["raw_shared_grad_norms"] = raw_task_grads.norm(dim=0).detach().cpu()
        extra_outputs["solver_shared_grad_norms"] = solver_task_grads.norm(dim=0).detach().cpu()
        return weighted_loss, extra_outputs

    def project_grad_update_to_tasks(self, grad_update, original_grads):
        """
        将FairGrad的grad_update投影回各个任务梯度方向
        """
        n_tasks = original_grads.shape[1]
        projected_grads = torch.zeros_like(original_grads)
        
        for i in range(n_tasks):
            task_grad = original_grads[:, i]
            task_norm = torch.norm(task_grad)
            
            if task_norm > 0:
                # 计算grad_update在task_grad方向上的投影
               # projection_coeff = torch.dot(grad_update, task_grad) / (task_norm ** 2)
              #  projected_grads[:, i] = projection_coeff * task_grad
                projected_grads[:, i] = torch.dot(grad_update, task_grad) / (task_norm)
            else:
                projected_grads[:, i] = torch.zeros_like(task_grad)
        
        return projected_grads

    def fairgrad(self, grads, alpha=1.0):
        GG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]

        x_start = np.ones(self.n_tasks) / self.n_tasks
        A = GG.data.cpu().numpy()

        def objfn(x):
            # return np.power(np.dot(A, x), alpha) - 1 / x
            return np.dot(A, x) - np.power(1 / x, 1 / alpha)

        res = least_squares(objfn, x_start, bounds=(0, np.inf))
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(grads.device)
        g = (grads * ww.view(1, -1)).sum(1)
        return g, GG.data.cpu().numpy(), w_cpu

    @staticmethod
    def grad2vec(shared_params, grads, grad_dims, task):
        # store the gradients
        grads[:, task].fill_(0.0)
        cnt = 0
        # for mm in m.shared_modules():
        #     for p in mm.parameters():

        for param in shared_params:
            grad = param.grad
            if grad is not None:
                grad_cur = grad.data.detach().clone()
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[: cnt + 1])
                grads[beg:en, task].copy_(grad_cur.data.view(-1))
            cnt += 1

    def overwrite_grad(self, shared_parameters, newgrad, grad_dims):
        newgrad = newgrad * self.n_tasks  # to match the sum loss
        cnt = 0

        # for mm in m.shared_modules():
        #     for param in mm.parameters():
        for param in shared_parameters:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[: cnt + 1])
            this_grad = newgrad[beg:en].contiguous().view(param.data.size())
            param.grad = this_grad.data.clone()
            cnt += 1

    def backward(
        self,
        losses: torch.Tensor,
        parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ):
        shared_parameters = self._to_parameter_list(shared_parameters)
        weighted_loss, extra_outputs = self.get_weighted_loss(
            losses, shared_parameters
        )
        torch.sum(losses).backward()
        if shared_parameters and self._latest_shared_grad is not None:
            self.overwrite_grad(
                shared_parameters,
                self._latest_shared_grad / self.n_tasks,
                [param.data.numel() for param in shared_parameters],
            )
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        return weighted_loss, extra_outputs


class FairGradOriginal(WeightMethod):
    def __init__(self, n_tasks, device: torch.device, alpha=1.0, max_norm=1.0):
        super().__init__(n_tasks, device=device)
        self.alpha = alpha
        self.max_norm = max_norm
        self.last_grads = None
        self.exp_avg = None
        self.step = 1
        self.grad_stats_history = []  # 添加这行
        self.mars_stats_history = []  # 添加这行

    def get_weighted_loss(
        self,
        losses,
        shared_parameters,
        **kwargs,
    ):
        """
        Parameters
        ----------
        losses :
        shared_parameters : shared parameters
        kwargs :
        Returns
        -------
        """
        # NOTE: we allow only shared params for now. Need to see paper for other options.
        grad_dims = []
        for param in shared_parameters:
            grad_dims.append(param.data.numel())
        grads = torch.Tensor(sum(grad_dims), self.n_tasks).to(self.device)

        for i in range(self.n_tasks):
            if i < self.n_tasks - 1:
                losses[i].backward(retain_graph=True)
            else:
                losses[i].backward()
            self.grad2vec(shared_parameters, grads, grad_dims, i)
            # multi_task_model.zero_grad_shared_modules()
            for p in shared_parameters:
                p.grad = None
             
        # # 计算处理前的梯度统计
        # grad_norm_before = torch.norm(grads)
        # grad_mean_before = torch.mean(grads)
        # grad_std_before = torch.std(grads)
        
        # # 计算每个任务的梯度方向（单位向量）
        # grad_directions_before = []
        # for i in range(self.n_tasks):
        #     task_grad = grads[:, i]
        #     task_norm = torch.norm(task_grad)
        #     if task_norm > 0:
        #         direction = task_grad / task_norm
        #     else:
        #         direction = torch.zeros_like(task_grad)
        #     grad_directions_before.append(direction)
        
        # 直接调用Vargrad处理梯度
        grads, self.last_grads, self.exp_avg = Vargrad(
            grads,
            last_grads=self.last_grads,
            exp_avg=self.exp_avg,
            step=self.step,
            beta=0.85,
            gamma=1.0,
            eps=1e-8
        )
        
        # 第二步：FairGrad处理（计算任务权重）
        grad_update, GTG, w_cpu = self.fairgrad(grads, alpha=self.alpha)
        
        # 第三步：将grad_update投影回各个任务梯度方向
      #  last_grads = self.project_grad_update_to_tasks(grad_update, grads)
        
        # # 计算处理后的梯度统计（MARS处理后的）
        # grad_norm_after = torch.norm(grads)
        # grad_mean_after = torch.mean(grads)
        # grad_std_after = torch.std(grads)
        
        # # 计算每个任务的梯度方向
        # grad_directions_after = []
        # for i in range(self.n_tasks):
        #     task_grad = grads[:, i]
        #     task_norm = torch.norm(task_grad)
        #     if task_norm > 0:
        #         direction = task_grad / task_norm
        #     else:
        #         direction = torch.zeros_like(task_grad)
        #     grad_directions_after.append(direction)
        
        # # 计算方向变化（余弦相似度）
        # direction_changes = []
        # for i in range(self.n_tasks):
        #     cos_sim = torch.dot(grad_directions_before[i], grad_directions_after[i])
        #     angle_change = torch.acos(torch.clamp(cos_sim, -1, 1)) * 180 / torch.pi
        #     direction_changes.append(angle_change.item())
        
        # # 打印统计信息
        # print(f"Step {self.step}:")
        # print(f"梯度范数: {grad_norm_before:.6f} -> {grad_norm_after:.6f} (ratio: {grad_norm_after/grad_norm_before:.4f})")
        # print(f"梯度均值: {grad_mean_before:.6f} -> {grad_mean_after:.6f}")
        # print(f"梯度标准差: {grad_std_before:.6f} -> {grad_std_after:.6f} (ratio: {grad_std_after/grad_std_before:.4f})")
        # print("方向变化角度:")
        # for i in range(self.n_tasks):
        #     print(f"  Task {i}: {direction_changes[i]:.2f}°")
        
        # # 保存数据用于绘图
        # grad_stats = {
        #     'step': self.step,
        #     'norm_before': grad_norm_before.item(),
        #     'norm_after': grad_norm_after.item(),
        #     'std_before': grad_std_before.item(),
        #     'std_after': grad_std_after.item(),
        #     'direction_changes': direction_changes,
        #     'task_norms_before': [torch.norm(grads[:, i]).item() for i in range(self.n_tasks)],
        #     'task_norms_after': [torch.norm(grads[:, i]).item() for i in range(self.n_tasks)]
        # }
        
        # # 添加到历史记录
        # self.grad_stats_history.append(grad_stats)
        
        self.step += 1
        
        # # 保存投影后的梯度作为last_grads
        # if hasattr(self, 'last_grads') and self.last_grads is not None:
        #     self.last_grads = last_grads.clone()
        # else:
        #     raise ValueError("last_grads is not initialized")
        
        self.overwrite_grad(shared_parameters, grad_update, grad_dims)
        return GTG, w_cpu

    def project_grad_update_to_tasks(self, grad_update, original_grads):
        """
        将FairGrad的grad_update投影回各个任务梯度方向
        """
        n_tasks = original_grads.shape[1]
        projected_grads = torch.zeros_like(original_grads)
        
        for i in range(n_tasks):
            task_grad = original_grads[:, i]
            task_norm = torch.norm(task_grad)
            
            if task_norm > 0:
                # 计算grad_update在task_grad方向上的投影
               # projection_coeff = torch.dot(grad_update, task_grad) / (task_norm ** 2)
              #  projected_grads[:, i] = projection_coeff * task_grad
                projected_grads[:, i] = torch.dot(grad_update, task_grad) / (task_norm)
            else:
                projected_grads[:, i] = torch.zeros_like(task_grad)
        
        return projected_grads

    def fairgrad(self, grads, alpha=1.0):
        GG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]

        x_start = np.ones(self.n_tasks) / self.n_tasks
        A = GG.data.cpu().numpy()

        def objfn(x):
            # return np.power(np.dot(A, x), alpha) - 1 / x
            return np.dot(A, x) - np.power(1 / x, 1 / alpha)

        res = least_squares(objfn, x_start, bounds=(0, np.inf))
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(grads.device)
        g = (grads * ww.view(1, -1)).sum(1)
        return g, GG.data.cpu().numpy(), w_cpu


    @staticmethod
    def grad2vec(shared_params, grads, grad_dims, task):
        # store the gradients
        grads[:, task].fill_(0.0)
        cnt = 0
        # for mm in m.shared_modules():
        #     for p in mm.parameters():

        for param in shared_params:
            grad = param.grad
            if grad is not None:
                grad_cur = grad.data.detach().clone()
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[: cnt + 1])
                grads[beg:en, task].copy_(grad_cur.data.view(-1))
            cnt += 1

    def overwrite_grad(self, shared_parameters, newgrad, grad_dims):
        newgrad = newgrad * self.n_tasks  # to match the sum loss
        cnt = 0

        # for mm in m.shared_modules():
        #     for param in mm.parameters():
        for param in shared_parameters:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[: cnt + 1])
            this_grad = newgrad[beg:en].contiguous().view(param.data.size())
            param.grad = this_grad.data.clone()
            cnt += 1

    def backward(
        self,
        losses: torch.Tensor,
        parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ):
        GTG, w = self.get_weighted_loss(losses, shared_parameters)
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        return None, {"GTG": GTG, "weights": w}  # NOTE: to align with all other weight methods


class FairGradOriginalComposable(FairGradOriginal):
    def __init__(
        self,
        n_tasks,
        device: torch.device,
        alpha=1.0,
        max_norm=1.0,
        preprocessing="vargrad",
        solver="fairgrad",
        scheduler="every_step",
        beta=0.85,
        psmgd_R=10,
        psmgd_alpha=0.5,
        psmgd_dynamic_threshold=1.0164316892623901,
        psmgd_dynamic_metric="refresh_rel_fro",
        psmgd_dynamic_direction="below",
    ):
        super().__init__(
            n_tasks=n_tasks,
            device=device,
            alpha=alpha,
            max_norm=max_norm,
        )
        if preprocessing not in ["identity", "vargrad"]:
            raise ValueError(f"unknown preprocessing {preprocessing}.")
        if solver != "fairgrad":
            raise ValueError(
                "FairGradOriginalComposable currently supports only solver=fairgrad."
            )
        if scheduler not in ["every_step", "psmgd_periodic", "psmgd_dynamic"]:
            raise ValueError(f"unknown scheduler {scheduler}.")
        if psmgd_dynamic_metric not in ["refresh_rel_fro", "step_rel_fro"]:
            raise ValueError(f"unknown psmgd_dynamic_metric {psmgd_dynamic_metric}.")
        if psmgd_dynamic_direction not in ["above", "below"]:
            raise ValueError(
                f"unknown psmgd_dynamic_direction {psmgd_dynamic_direction}."
            )

        self.preprocessing = preprocessing
        self.solver = solver
        self.scheduler = scheduler
        self.beta = beta
        self.gamma = 1.0
        self.psmgd_R = psmgd_R
        self.psmgd_alpha = psmgd_alpha
        self.psmgd_dynamic_threshold = psmgd_dynamic_threshold
        self.psmgd_dynamic_metric = psmgd_dynamic_metric
        self.psmgd_dynamic_direction = psmgd_dynamic_direction
        self.weights = torch.ones(n_tasks, device=device, dtype=torch.float32)
        self.last_candidate_weights = self.weights.clone()
        self.prev_solver_task_grads = None
        self.last_refresh_task_grads = None
        self.last_refresh_step = -1

    _scheduler_step = FairGrad._scheduler_step
    _compute_step_rel_fro = FairGrad._compute_step_rel_fro
    _compute_refresh_rel_fro = FairGrad._compute_refresh_rel_fro
    _compute_dynamic_refresh_score = FairGrad._compute_dynamic_refresh_score
    _should_call_solver = FairGrad._should_call_solver
    _apply_scheduler = FairGrad._apply_scheduler
    _build_u_telemetry = FairGrad._build_u_telemetry
    _to_parameter_list = staticmethod(FairGrad._to_parameter_list)
    _float = staticmethod(FairGrad._float)
    _float_list = staticmethod(FairGrad._float_list)

    def _collect_shared_task_grads(self, losses, shared_parameters):
        grad_dims = []
        for param in shared_parameters:
            grad_dims.append(param.data.numel())
        grads = torch.Tensor(sum(grad_dims), self.n_tasks).to(self.device)

        for i in range(self.n_tasks):
            if i < self.n_tasks - 1:
                losses[i].backward(retain_graph=True)
            else:
                losses[i].backward()
            self.grad2vec(shared_parameters, grads, grad_dims, i)
            for p in shared_parameters:
                p.grad = None

        return grads, grad_dims

    def _apply_preprocessing(self, raw_task_grads):
        if self.preprocessing != "vargrad":
            return raw_task_grads
        solver_task_grads, self.last_grads, self.exp_avg = Vargrad(
            raw_task_grads,
            last_grads=self.last_grads,
            exp_avg=self.exp_avg,
            step=self.step,
            beta=self.beta,
            gamma=self.gamma,
            eps=EPS,
        )
        return solver_task_grads

    def get_weighted_loss(
        self,
        losses,
        shared_parameters,
        **kwargs,
    ):
        shared_parameters = self._to_parameter_list(shared_parameters)
        extra_outputs = dict()
        if not shared_parameters:
            task_weights = self.weights.detach()
            weighted_loss = torch.sum(losses * task_weights)
            extra_outputs["weights"] = task_weights.detach().clone()
            extra_outputs["candidate_weights"] = self.last_candidate_weights.detach().clone()
            extra_outputs["updated_weights"] = False
            extra_outputs["solver_called"] = False
            extra_outputs["scheduler_step"] = self._scheduler_step()
            return weighted_loss, extra_outputs

        raw_task_grads, grad_dims = self._collect_shared_task_grads(
            losses, shared_parameters
        )
        solver_task_grads = self._apply_preprocessing(raw_task_grads)
        scheduler_step = self._scheduler_step()
        (
            solver_called,
            dynamic_refresh_triggered,
            dynamic_refresh_score,
        ) = self._should_call_solver(solver_task_grads, scheduler_step)

        if solver_called:
            _, GTG, w_cpu = self.fairgrad(solver_task_grads, alpha=self.alpha)
            candidate_weights = torch.from_numpy(w_cpu.astype(np.float32)).to(self.device)
            task_weights, updated_weights = self._apply_scheduler(candidate_weights)
        else:
            GTG = solver_task_grads.t().mm(solver_task_grads).detach().cpu().numpy()
            candidate_weights = self.last_candidate_weights
            task_weights = self.weights
            updated_weights = False

        grad_update = (
            solver_task_grads * task_weights.detach().view(1, -1)
        ).sum(dim=1)

        u_telemetry = self._build_u_telemetry(
            solver_task_grads=solver_task_grads,
            task_weights=task_weights,
            candidate_weights=self.last_candidate_weights,
            scheduler_step=scheduler_step,
            updated_weights=updated_weights,
            solver_called=solver_called,
            dynamic_refresh_triggered=dynamic_refresh_triggered,
            dynamic_refresh_score=dynamic_refresh_score,
        )
        self.step += 1
        self.overwrite_grad(shared_parameters, grad_update, grad_dims)

        weighted_loss = torch.sum(losses * task_weights.detach())
        extra_outputs["GTG"] = GTG
        extra_outputs["weights"] = task_weights.detach().clone()
        extra_outputs["candidate_weights"] = self.last_candidate_weights.detach().clone()
        extra_outputs["updated_weights"] = updated_weights
        extra_outputs["solver_called"] = solver_called
        extra_outputs["scheduler_step"] = scheduler_step
        extra_outputs["preprocessing"] = self.preprocessing
        extra_outputs["solver"] = self.solver
        extra_outputs["scheduler"] = self.scheduler
        extra_outputs["dynamic_refresh_metric"] = self.psmgd_dynamic_metric
        extra_outputs["dynamic_refresh_direction"] = self.psmgd_dynamic_direction
        extra_outputs["dynamic_refresh_score"] = float(dynamic_refresh_score)
        extra_outputs["dynamic_refresh_threshold"] = float(
            self.psmgd_dynamic_threshold
        )
        extra_outputs["dynamic_refresh_triggered"] = dynamic_refresh_triggered
        extra_outputs["u_telemetry"] = u_telemetry
        extra_outputs["raw_shared_grad_norms"] = raw_task_grads.norm(dim=0).detach().cpu()
        extra_outputs["solver_shared_grad_norms"] = solver_task_grads.norm(dim=0).detach().cpu()
        return weighted_loss, extra_outputs

    def backward(
        self,
        losses: torch.Tensor,
        parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ):
        shared_parameters = self._to_parameter_list(shared_parameters)
        _, extra_outputs = self.get_weighted_loss(losses, shared_parameters)
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        return None, extra_outputs


class GradDrop(WeightMethod):
    def __init__(self, n_tasks, device: torch.device, max_norm=1.0):
        super().__init__(n_tasks, device=device)
        self.max_norm = max_norm

    def get_weighted_loss(
        self,
        losses,
        shared_parameters,
        **kwargs,
    ):
        """
        Parameters
        ----------
        losses :
        shared_parameters : shared parameters
        kwargs :
        Returns
        -------
        """
        # NOTE: we allow only shared params for now. Need to see paper for other options.
        grad_dims = []
        for param in shared_parameters:
            grad_dims.append(param.data.numel())
        grads = torch.Tensor(sum(grad_dims), self.n_tasks).to(self.device)

        for i in range(self.n_tasks):
            if i < self.n_tasks:
                losses[i].backward(retain_graph=True)
            else:
                losses[i].backward()
            self.grad2vec(shared_parameters, grads, grad_dims, i)
            # multi_task_model.zero_grad_shared_modules()
            for p in shared_parameters:
                p.grad = None

        P = 0.5 * (1. + grads.sum(1) / (grads.abs().sum(1)+1e-8))
        U = torch.rand_like(grads[:,0])
        M = P.gt(U).view(-1,1)*grads.gt(0) + P.lt(U).view(-1,1)*grads.lt(0)
        g = (grads * M.float()).mean(1)
        self.overwrite_grad(shared_parameters, g, grad_dims)

    @staticmethod
    def grad2vec(shared_params, grads, grad_dims, task):
        # store the gradients
        grads[:, task].fill_(0.0)
        cnt = 0
        # for mm in m.shared_modules():
        #     for p in mm.parameters():

        for param in shared_params:
            grad = param.grad
            if grad is not None:
                grad_cur = grad.data.detach().clone()
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[: cnt + 1])
                grads[beg:en, task].copy_(grad_cur.data.view(-1))
            cnt += 1

    def overwrite_grad(self, shared_parameters, newgrad, grad_dims):
        newgrad = newgrad * self.n_tasks  # to match the sum loss
        cnt = 0

        # for mm in m.shared_modules():
        #     for param in mm.parameters():
        for param in shared_parameters:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[: cnt + 1])
            this_grad = newgrad[beg:en].contiguous().view(param.data.size())
            param.grad = this_grad.data.clone()
            cnt += 1

    def backward(
        self,
        losses: torch.Tensor,
        parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ):
        #GTG, w = self.get_weighted_loss(losses, shared_parameters)
        self.get_weighted_loss(losses, shared_parameters)
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        return None, None  # NOTE: to align with all other weight methods


class LOG_CAGrad(WeightMethod):
    def __init__(self, n_tasks, device: torch.device, c=0.4, max_norm=1.0):
        super().__init__(n_tasks, device=device)
        self.max_norm = max_norm
        self.c = c

    def get_weighted_loss(
        self,
        losses,
        shared_parameters,
        **kwargs,
    ):
        """
        Parameters
        ----------
        losses :
        shared_parameters : shared parameters
        kwargs :
        Returns
        -------
        """
        # NOTE: we allow only shared params for now. Need to see paper for other options.
        grad_dims = []
        for param in shared_parameters:
            grad_dims.append(param.data.numel())
        grads = torch.Tensor(sum(grad_dims), self.n_tasks).to(self.device)

        for i in range(self.n_tasks):
            if i < self.n_tasks:
                (losses[i].log()).backward(retain_graph=True)
            else:
                (losses[i].log()).backward()
            self.grad2vec(shared_parameters, grads, grad_dims, i)
            # multi_task_model.zero_grad_shared_modules()
            for p in shared_parameters:
                p.grad = None

        g, GTG, w_cpu = self.cagrad(grads, alpha=self.c, rescale=1)
        self.overwrite_grad(shared_parameters, g, grad_dims)
        #if self.max_norm > 0:
        #    torch.nn.utils.clip_grad_norm_(shared_parameters+task_specific_parameters, self.max_norm)
        return GTG, w_cpu

    def cagrad(self, grads, alpha=0.5, rescale=1):
        GG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]
        g0_norm = (GG.mean() + 1e-8).sqrt()  # norm of the average gradient

        x_start = np.ones(self.n_tasks) / self.n_tasks
        bnds = tuple((0, 1) for x in x_start)
        cons = {"type": "eq", "fun": lambda x: 1 - sum(x)}
        A = GG.numpy()
        b = x_start.copy()
        c = (alpha * g0_norm + 1e-8).item()

        def objfn(x):
            return (
                x.reshape(1, self.n_tasks).dot(A).dot(b.reshape(self.n_tasks, 1))
                + c
                * np.sqrt(
                    x.reshape(1, self.n_tasks).dot(A).dot(x.reshape(self.n_tasks, 1))
                    + 1e-8
                )
            ).sum()

        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(grads.device)
        gw = (grads * ww.view(1, -1)).sum(1)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm + 1e-8)
        g = grads.mean(1) + lmbda * gw
        if rescale == 0:
            return g, GG.numpy(), w_cpu
        elif rescale == 1:
            return g / (1 + alpha ** 2), GG.numpy(), w_cpu
        else:
            return g / (1 + alpha), GG.numpy(), w_cpu

    @staticmethod
    def grad2vec(shared_params, grads, grad_dims, task):
        # store the gradients
        grads[:, task].fill_(0.0)
        cnt = 0
        # for mm in m.shared_modules():
        #     for p in mm.parameters():

        for param in shared_params:
            grad = param.grad
            if grad is not None:
                grad_cur = grad.data.detach().clone()
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[: cnt + 1])
                grads[beg:en, task].copy_(grad_cur.data.view(-1))
            cnt += 1

    def overwrite_grad(self, shared_parameters, newgrad, grad_dims):
        newgrad = newgrad * self.n_tasks  # to match the sum loss
        cnt = 0

        # for mm in m.shared_modules():
        #     for param in mm.parameters():
        for param in shared_parameters:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[: cnt + 1])
            this_grad = newgrad[beg:en].contiguous().view(param.data.size())
            param.grad = this_grad.data.clone()
            cnt += 1

    def backward(
        self,
        losses: torch.Tensor,
        parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ):
        GTG, w = self.get_weighted_loss(losses, shared_parameters)
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        return None, {"GTG": GTG, "weights": w}  # NOTE: to align with all other weight methods


class RLW(WeightMethod):
    """Random loss weighting: https://arxiv.org/pdf/2111.10603.pdf"""

    def __init__(self, n_tasks, device: torch.device):
        super().__init__(n_tasks, device=device)

    def get_weighted_loss(self, losses: torch.Tensor, **kwargs):
        assert len(losses) == self.n_tasks
        weight = (F.softmax(torch.randn(self.n_tasks), dim=-1)).to(self.device)
        loss = torch.sum(losses * weight)

        return loss, dict(weights=weight)


class IMTLG(WeightMethod):
    """TOWARDS IMPARTIAL MULTI-TASK LEARNING: https://openreview.net/pdf?id=IMPnRXEWpvr"""

    def __init__(self, n_tasks, device: torch.device):
        super().__init__(n_tasks, device=device)

    def get_weighted_loss(
        self,
        losses,
        shared_parameters,
        **kwargs,
    ):
        grads = {}
        norm_grads = {}

        for i, loss in enumerate(losses):
            g = list(
                torch.autograd.grad(
                    loss,
                    shared_parameters,
                    retain_graph=True,
                )
            )
            grad = torch.cat([torch.flatten(grad) for grad in g])
            norm_term = torch.norm(grad)

            grads[i] = grad
            norm_grads[i] = grad / norm_term

        G = torch.stack(tuple(v for v in grads.values()))
        GTG = torch.mm(G, G.t())

        D = (
            G[
                0,
            ]
            - G[
                1:,
            ]
        )

        U = torch.stack(tuple(v for v in norm_grads.values()))
        U = (
            U[
                0,
            ]
            - U[
                1:,
            ]
        )
        first_element = torch.matmul(
            G[
                0,
            ],
            U.t(),
        )
        try:
            second_element = torch.inverse(torch.matmul(D, U.t()))
        except:
            # workaround for cases where matrix is singular
            second_element = torch.inverse(
                torch.eye(self.n_tasks - 1, device=self.device) * 1e-8
                + torch.matmul(D, U.t())
            )

        alpha_ = torch.matmul(first_element, second_element)
        alpha = torch.cat(
            (torch.tensor(1 - alpha_.sum(), device=self.device).unsqueeze(-1), alpha_)
        )

        loss = torch.sum(losses * alpha)
        extra_outputs = {}
        extra_outputs["weights"] = alpha
        extra_outputs["GTG"] = GTG.detach().cpu().numpy()
        return loss, extra_outputs


class LOG_IMTLG(WeightMethod):
    """TOWARDS IMPARTIAL MULTI-TASK LEARNING: https://openreview.net/pdf?id=IMPnRXEWpvr"""

    def __init__(self, n_tasks, device: torch.device):
        super().__init__(n_tasks, device=device)

    def get_weighted_loss(
        self,
        losses,
        shared_parameters,
        **kwargs,
    ):
        grads = {}
        norm_grads = {}

        for i, loss in enumerate(losses):
            g = list(
                torch.autograd.grad(
                    (loss + EPS).log(),
                    shared_parameters,
                    retain_graph=True,
                )
            )
            grad = torch.cat([torch.flatten(grad) for grad in g])
            norm_term = torch.norm(grad)

            grads[i] = grad
            norm_grads[i] = grad / norm_term

        G = torch.stack(tuple(v for v in grads.values()))
        GTG = torch.mm(G, G.t())

        D = (
            G[
                0,
            ]
            - G[
                1:,
            ]
        )

        U = torch.stack(tuple(v for v in norm_grads.values()))
        U = (
            U[
                0,
            ]
            - U[
                1:,
            ]
        )
        first_element = torch.matmul(
            G[
                0,
            ],
            U.t(),
        )
        try:
            second_element = torch.inverse(torch.matmul(D, U.t()))
        except:
            # workaround for cases where matrix is singular
            second_element = torch.inverse(
                torch.eye(self.n_tasks - 1, device=self.device) * 1e-8
                + torch.matmul(D, U.t())
            )

        alpha_ = torch.matmul(first_element, second_element)
        alpha = torch.cat(
            (torch.tensor(1 - alpha_.sum(), device=self.device).unsqueeze(-1), alpha_)
        )

        loss = torch.sum((losses + EPS).log() * alpha)
        extra_outputs = {}
        extra_outputs["weights"] = alpha
        extra_outputs["GTG"] = GTG.detach().cpu().numpy()
        return loss, extra_outputs


class DynamicWeightAverage(WeightMethod):
    """Dynamic Weight Average from `End-to-End Multi-Task Learning with Attention`.
    Modification of: https://github.com/lorenmt/mtan/blob/master/im2im_pred/model_segnet_split.py#L242
    """

    def __init__(
        self, n_tasks, device: torch.device, iteration_window: int = 25, temp=2.0
    ):
        """

        Parameters
        ----------
        n_tasks :
        iteration_window : 'iteration' loss is averaged over the last 'iteration_window' losses
        temp :
        """
        super().__init__(n_tasks, device=device)
        self.iteration_window = iteration_window
        self.temp = temp
        self.running_iterations = 0
        self.costs = np.ones((iteration_window * 2, n_tasks), dtype=np.float32)
        self.weights = np.ones(n_tasks, dtype=np.float32)

    def get_weighted_loss(self, losses, **kwargs):

        cost = losses.detach().cpu().numpy()

        # update costs - fifo
        self.costs[:-1, :] = self.costs[1:, :]
        self.costs[-1, :] = cost

        if self.running_iterations > self.iteration_window:
            ws = self.costs[self.iteration_window :, :].mean(0) / self.costs[
                : self.iteration_window, :
            ].mean(0)
            self.weights = (self.n_tasks * np.exp(ws / self.temp)) / (
                np.exp(ws / self.temp)
            ).sum()

        task_weights = torch.from_numpy(self.weights.astype(np.float32)).to(
            losses.device
        )
        loss = (task_weights * losses).mean()

        self.running_iterations += 1

        return loss, dict(weights=task_weights)


class WeightMethods:
    def __init__(self, method: str, n_tasks: int, device: torch.device, **kwargs):
        """
        :param method:
        """
        assert method in list(METHODS.keys()), f"unknown method {method}."

        self.method = METHODS[method](n_tasks=n_tasks, device=device, **kwargs)
        self.grad_stats_history = []  # 添加这个属性

    def get_weighted_loss(self, losses, **kwargs):
        return self.method.get_weighted_loss(losses, **kwargs)

    def backward(
        self, losses, **kwargs
    ) -> Tuple[Union[torch.Tensor, None], Union[Dict, None]]:
        return self.method.backward(losses, **kwargs)

    def __ceil__(self, losses, **kwargs):
        return self.backward(losses, **kwargs)

    def parameters(self):
        return self.method.parameters()
    
    # 添加属性访问方法
    @property
    def grad_stats_history(self):
        """访问内部方法的grad_stats_history属性"""
        if hasattr(self.method, 'grad_stats_history'):
            return self.method.grad_stats_history
        else:
            return []
    
    @grad_stats_history.setter
    def grad_stats_history(self, value):
        """设置内部方法的grad_stats_history属性"""
        if hasattr(self.method, 'grad_stats_history'):
            self.method.grad_stats_history = value


METHODS = dict(
    stl=STL,
    ls=LinearScalarization,
    uw=Uncertainty,
    scaleinvls=ScaleInvariantLinearScalarization,
    rlw=RLW,
    dwa=DynamicWeightAverage,

    pcgrad=PCGrad,
    mgda=MGDA,
    graddrop=GradDrop,
    log_mgda=LOG_MGDA,
    cagrad=CAGrad,
    log_cagrad=LOG_CAGrad,
    imtl=IMTLG,
    log_imtl=LOG_IMTLG,
    nashmtl=NashMTL,
    famo=FAMO,
    fairgrad=FairGrad,
    fairgrad_original=FairGradOriginal,
    fairgrad_original_composable=FairGradOriginalComposable,
)


def Vargrad(
    grads,
    last_grads=None,
    exp_avg=None,
    step=1,
    beta=0.95,
    gamma=1.0,
    eps=1e-8,
):
    """
    Vargrad: Variational gradient update for multi-task learning.
    
    Args:
        grads: Gradient matrix [grad_dim, n_tasks]
        last_grads: Previous gradients
        exp_avg: Exponential moving average of gradients
        step: Current step
        beta: First moment coefficient
        gamma: Gradient momentum coefficient
        eps: Small constant for numerical stability
        
    Returns:
        Updated gradients and momentum states
    """
    n_tasks = grads.shape[1]
    
    # Initialize states if None
    if last_grads is None:
        last_grads = torch.zeros_like(grads)
    if exp_avg is None:
        exp_avg = torch.zeros_like(grads)

    update = torch.zeros_like(grads)
    
    # Process each task
    for task_idx in range(n_tasks):
        task_grad = grads[:, task_idx]
        task_last_grad = last_grads[:, task_idx]
        task_exp_avg = exp_avg[:, task_idx]
        
        # Gradient momentum update
        c_t = task_grad + gamma * (beta / (1 - beta)) * (task_grad - task_last_grad)
        
        task_update = beta * task_exp_avg + (1 - beta) * c_t
        
        
        # Save results
        update[:, task_idx] = task_update
        exp_avg[:, task_idx] = task_update
        last_grads[:, task_idx] = task_grad.clone()

    return update, last_grads, exp_avg
