import torch
from transformers import AutoTokenizer, AutoModel


EMBED_MODEL_NAME = "jinaai/jina-embeddings-v3"
model = AutoModel.from_pretrained(EMBED_MODEL_NAME, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME, trust_remote_code=True)

if torch.cuda.is_available():
    model.cuda()


def embed(texts:list[str])->torch.Tensor:
    """
    Embed a text using the Jina model.
    """
    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    
    if torch.cuda.is_available():
        encoded = {key: tensor.cuda() for key, tensor in encoded.items()}
    with torch.no_grad():
        output = model(**encoded)

    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        embeddings = output.pooler_output
    else:
        last_hidden_state = output.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        mask_hidden = last_hidden_state * mask
        sum_hidden = mask_hidden.sum(dim=1)

        counts = mask.sum(dim=1).clamp(min=1e-9)
        embeddings = sum_hidden / counts

    embeddings = embeddings.cpu()
    return embeddings



    
        
