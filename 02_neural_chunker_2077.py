# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════╗
║  AETHERMIND TOKENIZER LAB — SCRIPT 2: NEURAL CHUNKER 2077      ║
║  Protótipo de tokenizer neural que aprende ONDE segmentar       ║
╚══════════════════════════════════════════════════════════════════╝

Este script implementa um Neural Chunker que:
1. Opera diretamente sobre bytes UTF-8 (sem regras de espaço)
2. Usa uma CNN 1D para aprender limites de segmentação
3. Treina com segmentações de referência do SentencePiece
4. Demonstra o conceito de "tokenizer aprendido" vs "tokenizer regrado"
"""
import os
import sys
import time
import json
import math
from pathlib import Path

BASE_DIR = Path(r"D:\AetherMind_Tokenizer_Lab")
CORPUS_PATH = BASE_DIR / "corpus" / "corpus_pt_br.txt"
RESULTS_DIR = BASE_DIR / "results"
NC_DIR = BASE_DIR / "neural_chunker"


def install_deps():
    """Instala dependências necessárias."""
    deps = ["torch", "sentencepiece"]
    for dep in deps:
        try:
            __import__(dep)
        except ImportError:
            print(f"[*] Instalando {dep}...")
            os.system(f'"{sys.executable}" -m pip install {dep} --quiet')


def text_to_bytes(text: str) -> list:
    """Converte texto para lista de byte IDs (0-255)."""
    return list(text.encode('utf-8'))


def bytes_to_text(byte_ids: list) -> str:
    """Converte lista de byte IDs de volta para texto."""
    return bytes(byte_ids).decode('utf-8', errors='replace')


def generate_training_data(corpus_path: str, sp_model_path: str):
    """
    Gera dados de treino para o Neural Chunker.

    Usa SentencePiece como "professor": onde o SP corta, marcamos
    como boundary=1. Onde não corta, boundary=0.

    Retorna lista de (byte_ids, boundary_labels).
    """
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=sp_model_path)

    with open(corpus_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]

    training_pairs = []

    for line in lines:
        # Tokeniza com SentencePiece
        pieces = sp.encode(line, out_type=str)

        # Reconstrói o texto token por token e marca boundaries
        byte_ids = text_to_bytes(line)
        boundaries = [0] * len(byte_ids)

        # Marca o último byte de cada token como boundary
        pos = 0
        for piece in pieces:
            # Remove o marcador de espaço do SentencePiece
            piece_clean = piece.replace('▁', ' ').lstrip()
            if piece.startswith('▁') and pos > 0:
                piece_bytes = (' ' + piece_clean).encode('utf-8')
            elif piece.startswith('▁') and pos == 0:
                piece_bytes = piece_clean.encode('utf-8')
            else:
                piece_bytes = piece.encode('utf-8')

            piece_len = len(piece_bytes)
            if pos + piece_len <= len(byte_ids):
                boundaries[pos + piece_len - 1] = 1  # Marca boundary no fim do token
            pos += piece_len

        # Garante que o último byte sempre é boundary
        if byte_ids:
            boundaries[-1] = 1

        training_pairs.append((byte_ids, boundaries))

    return training_pairs


def build_model():
    """Constrói o Neural Chunker."""
    import torch
    import torch.nn as nn

    class NeuralChunker(nn.Module):
        """
        Rede neural que aprende ONDE segmentar uma sequência de bytes.

        Arquitetura:
          Bytes → Embedding(256, 64) → Conv1D stack → Linear → Sigmoid

        Output: probabilidade de "corte" APÓS cada byte.

        Esta é uma rede LEVE — pode rodar em CPU ou GPU fraca.
        """
        def __init__(self, embed_dim=32, hidden_dim=64, n_layers=2, kernel_size=5):
            super().__init__()
            self.byte_embed = nn.Embedding(256, embed_dim)

            # Stack de convoluções com residual connections
            conv_layers = []
            in_dim = embed_dim
            for i in range(n_layers):
                out_dim = hidden_dim
                conv_layers.append(nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size,
                                            padding=kernel_size // 2))
                conv_layers.append(nn.ReLU())
                conv_layers.append(nn.Dropout(0.1))
                in_dim = out_dim
            self.conv_stack = nn.Sequential(*conv_layers)

            # Head de classificação binária: corte ou não-corte
            self.boundary_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 1),
            )

        def forward(self, byte_ids):
            """
            byte_ids: (batch, seq_len) — valores 0-255
            returns: (batch, seq_len) — probabilidades de corte
            """
            x = self.byte_embed(byte_ids)         # (batch, seq, embed_dim)
            x = x.transpose(1, 2)                 # (batch, embed_dim, seq) — para Conv1D
            x = self.conv_stack(x)                 # (batch, hidden, seq)
            x = x.transpose(1, 2)                 # (batch, seq, hidden)
            logits = self.boundary_head(x)         # (batch, seq, 1)
            return logits.squeeze(-1)              # (batch, seq)

    return NeuralChunker()


def train_neural_chunker(training_data, n_epochs=50, lr=0.001, batch_size=16):
    """
    Treina o Neural Chunker.
    """
    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pad_sequence

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[*] Device: {device}")

    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Class weights para lidar com desbalanceamento (mais 0s que 1s)
    total_0 = sum(b.count(0) for _, b in training_data)
    total_1 = sum(b.count(1) for _, b in training_data)
    pos_weight = torch.tensor([total_0 / max(total_1, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"[*] Modelo: {sum(p.numel() for p in model.parameters()):,} parâmetros")
    print(f"[*] Samples: {len(training_data)}")
    print(f"[*] Balance: {total_0} non-boundary / {total_1} boundary (ratio {total_0/max(total_1,1):.1f})")
    print(f"[*] Epochs: {n_epochs}")
    print()

    history = []

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        n_batches = 0

        # Shuffle
        import random
        random.shuffle(training_data)

        # Mini-batches
        for i in range(0, len(training_data), batch_size):
            batch = training_data[i:i+batch_size]

            # Pad sequences
            byte_seqs = [torch.tensor(b, dtype=torch.long) for b, _ in batch]
            label_seqs = [torch.tensor(l, dtype=torch.float) for _, l in batch]

            byte_padded = pad_sequence(byte_seqs, batch_first=True, padding_value=0).to(device)
            label_padded = pad_sequence(label_seqs, batch_first=True, padding_value=-1).to(device)

            # Forward
            logits = model(byte_padded)

            # Mask padding
            mask = label_padded >= 0
            logits_masked = logits[mask]
            labels_masked = label_padded[mask]

            loss = criterion(logits_masked, labels_masked)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(logits_masked) > 0.5).float()
            correct += (preds == labels_masked).sum().item()
            total += labels_masked.numel()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        accuracy = correct / max(total, 1) * 100
        history.append({"epoch": epoch + 1, "loss": avg_loss, "accuracy": accuracy})

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>3}/{n_epochs}  loss={avg_loss:.4f}  acc={accuracy:.1f}%")

    # Salvar modelo
    model_path = NC_DIR / "neural_chunker.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\n[✓] Modelo salvo: {model_path}")

    return model, device, history


def demo_neural_chunker(model, device):
    """
    Demonstra o Neural Chunker em ação.
    """
    import torch

    model.eval()

    test_sentences = [
        "O gato bebeu leite",
        "programação",
        "inteligência artificial",
        "O banco aprovou o empréstimo",
        "Sentei no banco da praça",
        "12345 + 67890",
        "A fórmula é softmax(QKᵀ/√d)V",
        "COVID mudou o mundo",
        "Mixture of Experts é poderosa",
    ]

    print(f"\n{'='*70}")
    print(f"  DEMO: NEURAL CHUNKER EM AÇÃO")
    print(f"{'='*70}")

    results = []

    for sentence in test_sentences:
        byte_ids = text_to_bytes(sentence)
        input_tensor = torch.tensor([byte_ids], dtype=torch.long).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.sigmoid(logits[0]).cpu().tolist()

        # Reconstruir chunks baseado nas boundaries preditas
        chunks = []
        current_chunk_bytes = []
        threshold = 0.5

        for j, (byte_val, prob) in enumerate(zip(byte_ids, probs)):
            current_chunk_bytes.append(byte_val)
            if prob > threshold:
                try:
                    chunk_text = bytes(current_chunk_bytes).decode('utf-8', errors='replace')
                except:
                    chunk_text = f"<{len(current_chunk_bytes)} bytes>"
                chunks.append(chunk_text)
                current_chunk_bytes = []

        # Último chunk
        if current_chunk_bytes:
            try:
                chunk_text = bytes(current_chunk_bytes).decode('utf-8', errors='replace')
            except:
                chunk_text = f"<{len(current_chunk_bytes)} bytes>"
            chunks.append(chunk_text)

        print(f"\n  Input: \"{sentence}\"")
        print(f"  Chunks ({len(chunks)}): {' | '.join(chunks)}")

        # Mostrar probabilidades por caractere (reconstruído)
        prob_display = []
        byte_pos = 0
        for char in sentence:
            char_bytes = char.encode('utf-8')
            # Pega a probabilidade do último byte do caractere
            last_byte_idx = byte_pos + len(char_bytes) - 1
            if last_byte_idx < len(probs):
                p = probs[last_byte_idx]
                if p > 0.7:
                    indicator = "█"
                elif p > 0.5:
                    indicator = "▓"
                elif p > 0.3:
                    indicator = "░"
                else:
                    indicator = "·"
                prob_display.append(f"{char}{indicator}")
            byte_pos += len(char_bytes)

        print(f"  Heatmap: {''.join(prob_display)}")
        print(f"  (█=corte forte  ▓=corte  ░=talvez  ·=sem corte)")

        results.append({
            "sentence": sentence,
            "chunks": chunks,
            "n_chunks": len(chunks),
        })

    return results


def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   ███╗   ██╗███████╗██╗   ██╗██████╗  █████╗ ██╗                ║
║   ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔══██╗██║                ║
║   ██╔██╗ ██║█████╗  ██║   ██║██████╔╝███████║██║                ║
║   ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██╔══██║██║                ║
║   ██║ ╚████║███████╗╚██████╔╝██║  ██║██║  ██║███████╗           ║
║   ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝           ║
║                                                                  ║
║          CHUNKER 2077 — Tokenizer Neural Adaptativo              ║
║          Aprende a segmentar texto como um Transformer           ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    install_deps()
    import torch

    # Verificar se o modelo SentencePiece existe (precisa rodar script 01 primeiro)
    sp_model = BASE_DIR / "models" / f"sp_unigram_1000.model"
    if not sp_model.exists():
        print(f"[!] Modelo SentencePiece não encontrado: {sp_model}")
        print(f"[!] Execute primeiro: python 01_train_and_compare.py")
        sys.exit(1)

    # Gerar dados de treino
    print("[*] Gerando dados de treino a partir do SentencePiece (professor)...")
    training_data = generate_training_data(str(CORPUS_PATH), str(sp_model))
    print(f"[✓] {len(training_data)} exemplos de treino gerados")

    # Treinar
    print("\n[*] Treinando Neural Chunker...")
    model, device, history = train_neural_chunker(training_data, n_epochs=60, lr=0.003, batch_size=8)

    # Demo
    nc_results = demo_neural_chunker(model, device)

    # Comparação Neural Chunker vs SentencePiece
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=str(sp_model))

    print(f"\n{'='*70}")
    print(f"  COMPARAÇÃO: NEURAL CHUNKER vs SENTENCEPIECE")
    print(f"{'='*70}")
    print(f"\n  {'FRASE':<35} {'NC':>4} {'SP':>4} {'MATCH':>6}")
    print(f"  {'─'*35} {'─'*4} {'─'*4} {'─'*6}")

    for r in nc_results:
        sp_pieces = sp.encode(r["sentence"], out_type=str)
        nc_n = r["n_chunks"]
        sp_n = len(sp_pieces)
        match = "≈" if abs(nc_n - sp_n) <= 1 else "≠"
        display = r["sentence"][:32] + "..." if len(r["sentence"]) > 35 else r["sentence"]
        print(f"  {display:<35} {nc_n:>4} {sp_n:>4} {match:>6}")

    # Salvar resultados
    results_path = RESULTS_DIR / "neural_chunker_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            "training_history": history,
            "chunker_results": nc_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[✓] Resultados salvos: {results_path}")

    # Salvar training history como texto legível
    history_path = RESULTS_DIR / "training_history.txt"
    with open(history_path, 'w', encoding='utf-8') as f:
        f.write("Epoch | Loss    | Accuracy\n")
        f.write("------+---------+---------\n")
        for h in history:
            f.write(f"{h['epoch']:>5} | {h['loss']:.4f}  | {h['accuracy']:.1f}%\n")
    print(f"[✓] Histórico salvo: {history_path}")

    print(f"\n{'='*70}")
    print(f"  NEURAL CHUNKER 2077 — EXECUÇÃO COMPLETA")
    print(f"  Modelo: {NC_DIR / 'neural_chunker.pt'}")
    print(f"  Resultados: {results_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
