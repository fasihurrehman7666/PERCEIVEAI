from fastapi import FastAPI, HTTPException
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import json
import os
import re
import difflib

app = FastAPI()

# 1. High-Accuracy Question-Aware BERT Grader Architecture (Matches Notebook)
class QABertWithProto(nn.Module):
    def __init__(self, num_questions: int, num_grades: int = 5, model_name='distilbert-base-uncased', proto_temp=0.07):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        H = self.encoder.config.hidden_size

        # Heads
        self.grade_head = nn.Linear(H, num_grades)     # standard head
        self.consistency_head = nn.Linear(H, 1)        # QA match/mismatch

        # In-model memory: question & grade embeddings -> prototypes
        self.E_q = nn.Embedding(num_questions, H)      # question embeddings
        self.E_g = nn.Embedding(num_grades, H)         # grade codes
        
        self.proto_temp = proto_temp

    def pooled(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        if getattr(out, 'pooler_output', None) is not None:
            return out.pooler_output
        last = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        # Weighted mean pooling as per notebook
        return (last * mask).sum(1) / mask.sum(1).clamp(min=1e-6)

    def prototypes_for_q(self, qids):
        # [B] -> [B,G,H]
        q_emb = self.E_q(qids)                   # [B,H]
        g_emb = self.E_g.weight.unsqueeze(0)     # [1,G,H]
        proto = F.normalize(q_emb.unsqueeze(1) + g_emb, dim=-1)  # [B,G,H]
        return proto

    def forward(self, input_ids, attention_mask, question_id):
        h = self.pooled(input_ids, attention_mask)         # [B,H]
        logits_head = self.grade_head(h)                  # [B,5]

        proto = self.prototypes_for_q(question_id)        # [B,5,H]
        h_norm = F.normalize(h, dim=-1).unsqueeze(1)      # [B,1,H]
        logits_proto = (h_norm * proto).sum(-1) / self.proto_temp  # [B,5]

        # Note: In training, cons_logit is calculated on [2B] inputs
        # In inference, we only pass the single input to consistency_head
        cons_logit = self.consistency_head(h).squeeze(-1) # [B]

        return h, logits_head, logits_proto, cons_logit

# 2. Initialization
model_loaded = False
try:
    # Load configuration and maps
    with open('grade_map.json') as f:
        GRADE2IDX = json.load(f)
        IDX2GRADE = {v: k for k, v in GRADE2IDX.items()}

    with open('qid_map.json') as f:
        qid_map = json.load(f)

    with open('config.json') as f:
        cfg = json.load(f)

    MODEL_NAME = cfg.get('model_name', 'distilbert-base-uncased')
    MAX_LEN = 256 # Match AnswerSet max_len from notebook
    CONS_THRESHOLD = 0.4 # More lenient gate
    HEAD_BLEND = cfg.get('head_blend', 0.5)
    PROTO_TEMP = cfg.get('proto_temp', 0.07)

    tokenizer = AutoTokenizer.from_pretrained(".")

    # Initialize model
    model = QABertWithProto(
        num_questions=len(qid_map),
        num_grades=len(GRADE2IDX),
        model_name=MODEL_NAME,
        proto_temp=PROTO_TEMP
    )

    # Load weights
    checkpoint = torch.load("best.pt", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    model_loaded = True
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    with open("load_error.log", "w") as f:
        f.write(str(e))

# Text normalization for robust matching
def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

normalized_qid_map = {normalize_text(k): v for k, v in qid_map.items()}

@app.post("/grade")
async def grade_answer(data: dict):
    if not model_loaded:
        raise HTTPException(status_code=500, detail="Model not loaded correctly")
    try:
        question = data.get('question', '')
        answer = data.get('answer', '')
        
        # 1. Get QID index with fuzzy matching
        norm_q = normalize_text(question)
        qid_idx = normalized_qid_map.get(norm_q)
        
        is_fuzzy = False
        if qid_idx is None:
            # Try fuzzy match
            matches = difflib.get_close_matches(norm_q, normalized_qid_map.keys(), n=1, cutoff=0.6)
            if matches:
                qid_idx = normalized_qid_map[matches[0]]
                is_fuzzy = True
                print(f"Fuzzy matched: '{question[:30]}...' -> '{matches[0][:30]}...'")
            else:
                print(f"WARNING: No match for question. Defaulting to index 0. Q: {question[:50]}...")
                qid_idx = 0
                is_known_question = False
            is_known_question = (qid_idx is not None)
        else:
            is_known_question = True

        qid_tensor = torch.tensor([qid_idx])

        # 2. Tokenize inputs exactly as in notebook
        text = f"Question: {question} [SEP] Answer: {answer}"
        enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=MAX_LEN)
        
        ids = enc['input_ids']
        att = enc['attention_mask']
        
        with torch.no_grad():
            h, lh, lp, cl = model(
                input_ids=ids,
                attention_mask=att,
                question_id=qid_tensor
            )
            
            # consistency score
            cons = torch.sigmoid(cl).item()

            # Blend head + prototype logits
            # Using additive blend as in evaluate()
            blended = lh + lp
            
            # Notebook logic:
            # if cons < thr: grade_idx = 4 # F
            if cons < CONS_THRESHOLD:
                grade_idx = 4  # F
                reason = "consistency_gate"
            else:
                grade_idx = int(blended.argmax(-1).item())
                reason = "model_prediction"
            
            probs = torch.softmax(blended, -1)[0].cpu().numpy().tolist()
            
            print(f"Inference: Q='{question[:30]}...' -> Grade={IDX2GRADE[grade_idx]} (cons={cons:.4f}, reason={reason})")

            return {
                "grade": IDX2GRADE[grade_idx],
                "confidence": round(float(torch.softmax(blended, -1)[0][grade_idx].item()), 4),
                "consistency_score": round(cons, 4),
                "is_consistent": cons >= CONS_THRESHOLD,
                "reason": reason,
                "is_known_question": is_known_question,
                "is_fuzzy": is_fuzzy,
                "probs": [round(float(x), 4) for x in probs],
                "debug": {
                    "qid_idx": qid_idx,
                    "lh_argmax": int(lh.argmax(-1).item()),
                    "lp_argmax": int(lp.argmax(-1).item()),
                    "cons_logit": float(cl.item())
                }
            }
            
    except Exception as e:
        print(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
