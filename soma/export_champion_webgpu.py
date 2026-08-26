#!/usr/bin/env python3
"""Export the fixed-1B N=8 SetAttn-EMA champion as one-token INT4 ONNX graphs.

Each expert and the combiner are separate static-shape graphs. This keeps every
artifact below GitHub's 100 MB file limit and lets ORT Web retain recurrent state
on WebGPU between token steps.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

V, N, DE, DT, DC = 256, 8, 128, 1922, 1072

def linear(x, w, b=None):
    """ONNX MatMul(+Add), rather than Gemm, so MatMulNBits can quantize it."""
    y = torch.matmul(x, w.transpose(-1, -2))
    return y if b is None else y + b


class ExpertStep(nn.Module):
    def __init__(self, state):
        super().__init__()
        self.embed = nn.Parameter(state["embed.weight"].float(), requires_grad=False)
        self.in_proj = nn.Parameter(state["in_proj.weight"].float(), requires_grad=False)
        self.blocks = nn.ModuleList()
        for i in range(2):
            b = nn.ParameterDict({
                "ln1": nn.Parameter(state[f"blocks.{i}.ln1.weight"].float(), requires_grad=False),
                "wih": nn.Parameter(state[f"blocks.{i}.lstm.weight_ih_l0"].float(), requires_grad=False),
                "whh": nn.Parameter(state[f"blocks.{i}.lstm.weight_hh_l0"].float(), requires_grad=False),
                "ln2": nn.Parameter(state[f"blocks.{i}.ln2.weight"].float(), requires_grad=False),
                "fc1": nn.Parameter(state[f"blocks.{i}.fc1.weight"].float(), requires_grad=False),
                "fc2": nn.Parameter(state[f"blocks.{i}.fc2.weight"].float(), requires_grad=False),
            })
            self.blocks.append(b)
        self.final_norm = nn.Parameter(state["final_norm.weight"].float(), requires_grad=False)
        self.out_proj = nn.Parameter(state["out_proj.weight"].float(), requires_grad=False)

    @staticmethod
    def norm(x, gain):
        return F.layer_norm(x, (x.shape[-1],), gain, None, 1e-5)

    def forward(self, token, h0, c0, h1, c1):
        x = linear(F.embedding(token, self.embed), self.in_proj)
        hs, cs = [], []
        for i, b in enumerate(self.blocks):
            h, c = (h0, c0) if i == 0 else (h1, c1)
            u = self.norm(x, b["ln1"])
            g = linear(u, b["wih"]) + linear(h, b["whh"])
            ig, fg, gg, og = g.chunk(4, -1)
            cn = torch.sigmoid(fg) * c + torch.sigmoid(ig) * torch.tanh(gg)
            hn = torch.sigmoid(og) * torch.tanh(cn)
            x = x + hn
            x = x + linear(F.gelu(linear(self.norm(x, b["ln2"]), b["fc1"])), b["fc2"])
            hs.append(hn); cs.append(cn)
        rep = linear(self.norm(x, self.final_norm), self.out_proj)
        logp = F.log_softmax(linear(rep, self.embed), -1)
        return logp, rep, hs[0], cs[0], hs[1], cs[1]


class CombinerStep(nn.Module):
    def __init__(self, sd):
        super().__init__()
        # Preserve checkpoint names to make auditing against the trainer trivial.
        self.eid = nn.Parameter(sd["eid"].float(), requires_grad=False)
        self.eta = nn.Parameter(sd["eta"].float(), requires_grad=False)
        self.adapt_w = nn.Parameter(torch.stack([sd[f"adapt.{i}.weight"].float() for i in range(N)]), requires_grad=False)
        self.adapt_b = nn.Parameter(torch.stack([sd[f"adapt.{i}.bias"].float() for i in range(N)]), requires_grad=False)
        self.layers = nn.ModuleList()
        for i in range(2):
            p = f"setenc.layers.{i}."
            d = nn.ParameterDict({k: nn.Parameter(sd[p + src].float(), requires_grad=False) for k, src in {
                "qkv_w":"self_attn.in_proj_weight", "qkv_b":"self_attn.in_proj_bias",
                "attn_w":"self_attn.out_proj.weight", "attn_b":"self_attn.out_proj.bias",
                "ff1_w":"linear1.weight", "ff1_b":"linear1.bias",
                "ff2_w":"linear2.weight", "ff2_b":"linear2.bias",
                "n1_w":"norm1.weight", "n1_b":"norm1.bias",
                "n2_w":"norm2.weight", "n2_b":"norm2.bias"}.items()})
            self.layers.append(d)
        self.cemb = nn.Parameter(sd["cemb.weight"].float(), requires_grad=False)
        self.ctx = nn.ModuleList()
        for i in range(2):
            p = f"ctx.layers.{i}."
            self.ctx.append(nn.ParameterDict({
                "wih": nn.Parameter(sd[p+"weight_ih_l0"].float(), requires_grad=False),
                "whh": nn.Parameter(sd[p+"weight_hh_l0"].float(), requires_grad=False),
                "b": nn.Parameter((sd[p+"bias_ih_l0"] + sd[p+"bias_hh_l0"]).float(), requires_grad=False),
            }))
        self.qw = nn.Parameter(sd["q.weight"].float(), requires_grad=False)
        self.kw = nn.Parameter(sd["k.weight"].float(), requires_grad=False)
        self.dw = nn.Parameter(sd["delta.weight"].float(), requires_grad=False)
        self.db = nn.Parameter(sd["delta.bias"].float(), requires_grad=False)

    @staticmethod
    def ln(x, w, b): return F.layer_norm(x, (DC,), w, b, 1e-5)

    @staticmethod
    def lstm(x, h, c, p):
        g = linear(x, p["wih"], p["b"]) + linear(h, p["whh"])
        i, f, z, o = g.chunk(4, -1)
        cn = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(z)
        hn = torch.sigmoid(o) * torch.tanh(cn)
        return hn, cn

    def set_layer(self, x, p):
        z = self.ln(x, p["n1_w"], p["n1_b"])
        q, k, v = linear(z, p["qkv_w"], p["qkv_b"]).chunk(3, -1)
        q = q.reshape(N, 4, DC//4).transpose(0, 1)
        k = k.reshape(N, 4, DC//4).transpose(0, 1)
        v = v.reshape(N, 4, DC//4).transpose(0, 1)
        a = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(DC//4), -1)
        z = torch.matmul(a, v).transpose(0, 1).reshape(N, DC)
        x = x + linear(z, p["attn_w"], p["attn_b"])
        z = self.ln(x, p["n2_w"], p["n2_b"])
        return x + linear(F.gelu(linear(z, p["ff1_w"], p["ff1_b"])), p["ff2_w"], p["ff2_b"])

    def forward(self, token, logp, rep, prev_logp, ema_num, ema_den, h0, c0, h1, c1):
        # Competence at this position is based only on already-observed outcomes.
        snll = -prev_logp[:, token[0]]
        ema = ema_num / torch.clamp_min(ema_den, 1.0)
        p = torch.exp(logp)
        ent = -(p * logp).sum(-1)
        top = torch.topk(p, 2, -1).values
        feat = torch.cat((rep, ent[:,None], top[:,0,None], (top[:,0]-top[:,1])[:,None], snll[:,None], ema.transpose(0,1)), -1)
        u = torch.stack([linear(feat[i], self.adapt_w[i], self.adapt_b[i]) for i in range(N)]) + self.eid
        for layer in self.layers: u = self.set_layer(u, layer)

        h0n, c0n = self.lstm(F.embedding(token, self.cemb), h0, c0, self.ctx[0])
        ctx, c1n = self.lstm(h0n, h1, c1, self.ctx[1])
        score = (linear(ctx, self.qw) * linear(u, self.kw)).sum(-1) / math.sqrt(DC)
        score = score - ema[1] + self.eta
        w = torch.softmax(score, -1)
        base = torch.log(torch.clamp_min((w[:,None] * p).sum(0), 1e-9))
        out = F.log_softmax(base + linear(ctx, self.dw, self.db).squeeze(0), -1)

        rho = torch.tensor([.9,.99,.997], dtype=ema_num.dtype)[:,None]
        num_n = rho * ema_num + snll[None,:]
        den_n = rho * ema_den + 1.0
        return out, w, logp, num_n, den_n, h0n, c0n, ctx, c1n


def export(module, inputs, path, input_names, output_names):
    module.eval()
    torch.onnx.export(module, inputs, path, input_names=input_names, output_names=output_names,
                      opset_version=18, dynamo=False, do_constant_folding=True,
                      keep_initializers_as_inputs=False)


def quantize(src: Path, dst: Path):
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        HQQWeightOnlyQuantConfig, MatMulNBitsQuantizer)
    # HQQ materially preserves the recurrent trajectory better than naive
    # round-to-nearest INT4 at the same block size.
    cfg = HQQWeightOnlyQuantConfig(block_size=64, bits=4)
    q = MatMulNBitsQuantizer(str(src), algo_config=cfg)
    q.process()
    q.model.save_model_to_file(str(dst), use_external_data_format=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", default="/home/claudeuser/sft_expert_cache")
    ap.add_argument("--combiner", default="/home/claudeuser/soma_web_champion/setattn_ema.complete.pt")
    ap.add_argument("--out", default="champion")
    ap.add_argument("--only", choices=["all","combiner",*[f"e{i}" for i in range(N)]], default="all")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    z = torch.zeros(1, DT)
    if a.only in ("all", "combiner"):
        ck = torch.load(a.combiner, map_location="cpu", weights_only=False)
        m = CombinerStep(ck["model"])
        inp = (torch.zeros(1,dtype=torch.long), torch.full((N,V),-math.log(V)), torch.zeros(N,DE),
               torch.full((N,V),-math.log(V)), torch.zeros(3,N), torch.zeros(3,1),
               torch.zeros(1,DC),torch.zeros(1,DC),torch.zeros(1,DC),torch.zeros(1,DC))
        fp=out/"combiner.fp32.onnx"; qp=out/"combiner.int4.onnx"
        export(m,inp,fp,["token","logp","rep","prev_logp","ema_num","ema_den","h0","c0","h1","c1"],
               ["out","gate","prev_logp_n","ema_num_n","ema_den_n","h0_n","c0_n","h1_n","c1_n"])
        quantize(fp,qp)
    for i in range(N):
        if a.only not in ("all",f"e{i}"): continue
        p=Path(a.experts)/f"n8_r05_e128_dt1922_c{i}.best.pt"
        ck=torch.load(p,map_location="cpu",weights_only=False); m=ExpertStep(ck["model"])
        fp=out/f"expert{i}.fp32.onnx"; qp=out/f"expert{i}.int4.onnx"
        export(m,(torch.zeros(1,dtype=torch.long),z,z,z,z),fp,["token","h0","c0","h1","c1"],
               ["logp","rep","h0_n","c0_n","h1_n","c1_n"])
        quantize(fp,qp)
    manifest={"name":"SOMA 1B + SetAttn-EMA","N":8,"params":999766200,"val_loss":0.8598677783675207,
              "quantization":"HQQ INT4, block 64","experts":[f"expert{i}.int4.onnx" for i in range(N)],
              "combiner":"combiner.int4.onnx"}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")

if __name__ == "__main__": main()
