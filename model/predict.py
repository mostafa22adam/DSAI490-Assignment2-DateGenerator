import argparse
import os
import torch
import torch.nn as nn
import calendar
from datetime import date as date_cls, timedelta

DAY_TOKENS    = ["[MON]","[TUE]","[WED]","[THU]","[FRI]","[SAT]","[SUN]"]
MONTH_TOKENS  = ["[JAN]","[FEB]","[MAR]","[APR]","[MAY]","[JUN]",
                 "[JUL]","[AUG]","[SEP]","[OCT]","[NOV]","[DEC]"]
LEAP_TOKENS   = ["[False]","[True]"]
DECADE_TOKENS = [f"[{d}]" for d in range(180, 221)]
CONDITION_VOCAB = DAY_TOKENS + MONTH_TOKENS + LEAP_TOKENS + DECADE_TOKENS
OUTPUT_VOCAB    = ["<PAD>","<SOS>","<EOS>"] + list("0123456789-")
cond_to_idx = {t: i for i, t in enumerate(CONDITION_VOCAB)}
out_to_idx  = {t: i for i, t in enumerate(OUTPUT_VOCAB)}
idx_to_out  = {i: t for t, i in out_to_idx.items()}
PAD_IDX = out_to_idx["<PAD>"]
SOS_IDX = out_to_idx["<SOS>"]
EOS_IDX = out_to_idx["<EOS>"]
COND_VOCAB_SIZE = len(CONDITION_VOCAB)
OUT_VOCAB_SIZE  = len(OUTPUT_VOCAB)
DAY_MAP   = {"MON":0,"TUE":1,"WED":2,"THU":3,"FRI":4,"SAT":5,"SUN":6}
MONTH_MAP = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
             "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}

class TransformerDateGenerator(nn.Module):
    def __init__(self, emb_dim=64, nhead=4, num_enc_layers=2, num_dec_layers=2,
                 ff_dim=256, dropout=0.1, max_seq_len=12):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.cond_emb  = nn.Embedding(COND_VOCAB_SIZE, emb_dim)
        self.date_emb  = nn.Embedding(OUT_VOCAB_SIZE, emb_dim, padding_idx=PAD_IDX)
        self.pos_emb   = nn.Embedding(max_seq_len + 2, emb_dim)
        self.encoder   = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(emb_dim, nhead, ff_dim, dropout, batch_first=True),
            num_layers=num_enc_layers)
        self.decoder   = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(emb_dim, nhead, ff_dim, dropout, batch_first=True),
            num_layers=num_dec_layers)
        self.output_fc = nn.Linear(emb_dim, OUT_VOCAB_SIZE)

    def encode(self, conds):
        return self.encoder(self.cond_emb(conds))

    def decode_step(self, tgt_ids, memory):
        T   = tgt_ids.size(1)
        pos = torch.arange(T, device=tgt_ids.device).unsqueeze(0)
        x   = self.date_emb(tgt_ids) + self.pos_emb(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=tgt_ids.device)
        out  = self.decoder(x, memory, tgt_mask=mask,
                            tgt_key_padding_mask=(tgt_ids == PAD_IDX))
        return self.output_fc(out)

    @torch.no_grad()
    def generate(self, conds, device, max_len=12):
        self.eval()
        memory = self.encode(conds)
        ys     = torch.full((conds.size(0), 1), SOS_IDX, dtype=torch.long, device=device)
        for _ in range(max_len):
            next_id = self.decode_step(ys, memory)[:, -1].argmax(dim=-1, keepdim=True)
            ys      = torch.cat([ys, next_id], dim=1)
            if (next_id.squeeze(-1) == EOS_IDX).all():
                break
        results = []
        for i in range(conds.size(0)):
            chars = []
            for idx in ys[i, 1:].tolist():
                if idx == EOS_IDX: break
                if idx not in (SOS_IDX, PAD_IDX): chars.append(idx_to_out[idx])
            results.append("".join(chars))
        return results

def encode_conditions(day, month, leap, decade):
    return torch.tensor([cond_to_idx[t] for t in [day, month, leap, decade]], dtype=torch.long)

def parse_date(s):
    try:
        p = s.strip().split("-")
        return date_cls(int(p[2]), int(p[1]), int(p[0]))
    except: return None

def check_all(date_str, day_tok, month_tok, leap_tok, decade_tok):
    dt = parse_date(date_str)
    if dt is None or not (date_cls(1800,1,1) <= dt <= date_cls(2200,12,31)): return False
    return (dt.weekday() == DAY_MAP[day_tok.strip("[]")] and
            dt.month == MONTH_MAP[month_tok.strip("[]")] and
            calendar.isleap(dt.year) == (leap_tok.strip("[]") == "True") and
            dt.year // 10 == int(decade_tok.strip("[]")))

def fix_day(date_str, day_tok, month_tok, leap_tok, decade_tok):
    if check_all(date_str, day_tok, month_tok, leap_tok, decade_tok): return date_str
    dt = parse_date(date_str)
    if dt is None: return date_str
    for delta in range(1, 400):
        for sign in [1, -1]:
            cand = dt + timedelta(days=delta * sign)
            s    = f"{cand.day}-{cand.month}-{cand.year}"
            if check_all(s, day_tok, month_tok, leap_tok, decade_tok): return s
    return date_str

def find_weights():
    base = os.path.dirname(os.path.abspath(__file__))
    for p in [os.path.join(base, "transformer", "transformer_weights.pth"),
              os.path.join(base, "transformer_weights.pth"),
              "/content/transformer_weights.pth"]:
        if os.path.exists(p): return p
    raise FileNotFoundError("Cannot find transformer_weights.pth")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True)
    parser.add_argument("-o", required=True)
    args   = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = TransformerDateGenerator().to(device)
    model.load_state_dict(torch.load(find_weights(), map_location=device))
    model.eval()
    with open(args.i) as f:
        lines = [l.strip() for l in f if l.strip()]
    results = []
    for start in range(0, len(lines), 64):
        batch  = lines[start:start+64]
        toks   = [l.split() for l in batch]
        conds  = torch.stack([encode_conditions(*t[:4]) for t in toks]).to(device)
        preds  = model.generate(conds, device)
        for t, pred in zip(toks, preds):
            fixed = fix_day(pred, t[0], t[1], t[2], t[3])
            results.append(f"{t[0]} {t[1]} {t[2]} {t[3]} {fixed}")
    with open(args.o, "w") as f:
        f.write("\n".join(results) + "\n")
    print(f"Done. {len(results)} predictions written to {args.o}")

if __name__ == "__main__":
    main()