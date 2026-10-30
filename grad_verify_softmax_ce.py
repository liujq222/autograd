# -*- coding: utf-8 -*-
"""
Softmax + CrossEntropy 梯度验证
验证目标：对 z 求损失 L 的梯度应等于 (p - y)，即
    dL/dz_i = p_i - y_i
三种方式核对：
  1) 手推解析式    grad_analytic = p - y
  2) PyTorch autograd（自动微分）
  3) 数值有限差分（中心差分，验证用）  (L(z+εe_i)-L(z-εe_i)) / (2ε)
误差全部记录进笔记。
"""
import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)

# ---------- 稳定 softmax ----------
def softmax_stable(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=-1, keepdims=True)   # log-sum-exp 稳定
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)

def cross_entropy_np(p: np.ndarray, y: np.ndarray) -> float:
    """p: softmax 输出概率; y: one-hot 标签; 返回标量损失"""
    eps = 1e-12
    return -np.sum(y * np.log(np.clip(p, eps, 1.0)))

# ---------- 1) 解析梯度 ----------
z = np.array([2.0, -1.0, 0.5, 3.0, 1.0])     # 未归一化 logits
y = np.array([0.0, 0.0, 1.0, 0.0, 0.0])      # one-hot: 类别 2 为真
p = softmax_stable(z)
grad_analytic = p - y                         # 理论值 dL/dz = p - y
loss_np = cross_entropy_np(p, y)

print("logits z :", z)
print("标签 y(onehot):", y)
print("softmax p:", np.round(p, 6))
print("损失 L:", round(loss_np, 6))
print("解析梯度 p-y:", np.round(grad_analytic, 6))

# ---------- 2) PyTorch autograd ----------
zt = torch.tensor(z, dtype=torch.double, requires_grad=True)
yt = torch.tensor(y, dtype=torch.double)
logp = F.log_softmax(zt, dim=0)
loss_t = -(yt * logp).sum()
loss_t.backward()
grad_autograd = zt.grad.detach().numpy().copy()

print("\n[2] autograd 梯度:", np.round(grad_autograd, 6))

# ---------- 3) 数值有限差分（中心差分） ----------
def loss_of(z_val: np.ndarray) -> float:
    return cross_entropy_np(softmax_stable(z_val), y)

eps = 1e-5
grad_fd = np.zeros_like(z)
for i in range(z.size):
    zp = z.copy(); zp[i] += eps
    zm = z.copy(); zm[i] -= eps
    grad_fd[i] = (loss_of(zp) - loss_of(zm)) / (2 * eps)

print("[3] 有限差分梯度 :", np.round(grad_fd, 6))

# ---------- 误差统计 ----------
err_auto = np.abs(grad_autograd - grad_analytic).max()
err_fd   = np.abs(grad_fd - grad_analytic).max()
rel_fd   = np.linalg.norm(grad_fd - grad_analytic) / (np.linalg.norm(grad_analytic) + 1e-12)

print("\n========== 误差核对 ==========")
print("解析 vs autograd 最大绝对误差 : {:.2e}".format(err_auto))
print("解析 vs 有限差分 最大绝对误差 : {:.2e}  (eps={})".format(err_fd, eps))
print("解析 vs 有限差分 相对L2误差   : {:.2e}".format(rel_fd))

# ---------- 验证结论 ----------
def verdict(err, tol=1e-4):
    return "PASS ✓" if err < tol else "FAIL ✗"
print("\nautograd 核对 :", verdict(err_auto))
print("有限差分核对   :", verdict(err_fd))

# 一行签名打印，便于记入笔记
print("\n[NOTE] grad = p - y, |auto|=%.2e, |fd|=%.2e, rel=%.2e" % (err_auto, err_fd, rel_fd))
#（注：内容由AI生成）
