# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════╗
║  AETHERMIND TOKENIZER LAB — SCRIPT 1: TREINO & COMPARAÇÃO      ║
║  Treina SentencePiece em modo BPE e Unigram, compara resultados ║
╚══════════════════════════════════════════════════════════════════╝
"""
import os
import sys
import json
import time
from pathlib import Path
from collections import Counter

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = Path(r"D:\AetherMind_Tokenizer_Lab")
CORPUS_PATH = BASE_DIR / "corpus" / "corpus_pt_br.txt"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
VOCAB_SIZE = 1000  # Ajustado ao tamanho do corpus (141 linhas, ~10K chars)

# Frases de teste para comparação
TEST_SENTENCES = [
    "O gato bebeu leite",
    "programação",
    "programador",
    "programando",
    "programar",
    "inteligência artificial",
    "Tokenização é essencial para LLMs",
    "O banco aprovou o empréstimo",
    "Sentei no banco da praça",
    "COVID-19 mudou o mundo em 2020",
    "12345 + 67890 = 80235",
    "こんにちは mundo",  # Japonês + Português
    "A fórmula é softmax(QKᵀ/√d)V",
    "O Ruan está estudando machine learning",
    "Mixture of Experts é uma arquitetura poderosa",
    "Emaranhamento quântico conecta partículas",
    "O café brasileiro é reconhecido mundialmente",
    "Eu amo programar em Python e Rust",
]


def fix_mangled_string(s):
    """Fix Mojibake from PowerShell encoding issues."""
    for enc in ['cp1252', 'cp850', 'cp437', 'iso-8859-1']:
        try:
            return s.encode(enc).decode('utf-8')
        except UnicodeError:
            continue
    return s


def install_sentencepiece():
    """Instala sentencepiece se não estiver disponível."""
    try:
        import sentencepiece
        return True
    except ImportError:
        print("[*] Instalando sentencepiece...")
        os.system(f'"{sys.executable}" -m pip install sentencepiece --quiet')
        try:
            import sentencepiece
            return True
        except ImportError:
            print("[!] FALHA ao instalar sentencepiece.")
            return False


def train_tokenizer(model_type: str, vocab_size: int) -> str:
    """
    Treina um tokenizer SentencePiece.
    model_type: 'bpe' ou 'unigram'
    Retorna o caminho do modelo treinado.
    """
    import sentencepiece as spm

    model_prefix = str(MODELS_DIR / f"sp_{model_type}_{vocab_size}")

    print(f"\n{'='*60}")
    print(f"  TREINANDO: {model_type.upper()} | vocab_size={vocab_size}")
    print(f"{'='*60}")

    start = time.time()

    spm.SentencePieceTrainer.train(
        input=str(CORPUS_PATH),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=0.9995,
        num_threads=os.cpu_count() or 4,
        byte_fallback=True,                    # Nunca gera <unk>
        split_digits=True,                     # Separa dígitos individualmente
        allow_whitespace_only_pieces=False,
        max_sentence_length=4096,
        shuffle_input_sentence=True,
        # Unigram-specific: shrinking factor
        shrinking_factor=0.75 if model_type == 'unigram' else 0.75,
    )

    elapsed = time.time() - start
    model_file = f"{model_prefix}.model"
    vocab_file = f"{model_prefix}.vocab"

    print(f"  ✓ Modelo treinado em {elapsed:.2f}s")
    print(f"  ✓ Arquivo: {model_file}")
    print(f"  ✓ Vocab:   {vocab_file}")

    return model_file


def analyze_tokenizer(model_path: str, model_type: str):
    """
    Análise completa de um tokenizer treinado.
    """
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=model_path)

    print(f"\n{'='*60}")
    print(f"  ANÁLISE: {model_type.upper()}")
    print(f"{'='*60}")
    print(f"  Vocab size: {sp.get_piece_size()}")

    # ── Tokenização das frases de teste ──
    results = []
    total_tokens = 0

    print(f"\n  {'FRASE':<45} {'TOKENS':>6}  SEGMENTAÇÃO")
    print(f"  {'─'*45} {'─'*6}  {'─'*40}")

    for sentence in TEST_SENTENCES:
        # Segmentação determinística
        pieces = sp.encode(sentence, out_type=str)
        ids = sp.encode(sentence, out_type=int)
        n_tokens = len(pieces)
        total_tokens += n_tokens

        display = sentence[:42] + "..." if len(sentence) > 45 else sentence
        pieces_str = " | ".join(pieces)
        print(f"  {display:<45} {n_tokens:>6}  {pieces_str}")

        results.append({
            "sentence": sentence,
            "tokens": pieces,
            "ids": ids,
            "n_tokens": n_tokens,
        })

    avg_tokens = total_tokens / len(TEST_SENTENCES)
    print(f"\n  MÉDIA de tokens por frase: {avg_tokens:.1f}")

    # ── Subword Regularization (só funciona bem com Unigram) ──
    if model_type == "unigram":
        print(f"\n  {'─'*60}")
        print(f"  SUBWORD REGULARIZATION (5 amostragens de 'programação'):")
        for i in range(5):
            pieces = sp.encode("programação", out_type=str,
                             enable_sampling=True, alpha=0.1, nbest_size=-1)
            print(f"    Amostra {i+1}: {' | '.join(pieces)}")

    # ── Análise de vocabulário ──
    print(f"\n  {'─'*60}")
    print(f"  TOP 30 tokens do vocabulário:")
    vocab_pieces = []
    for i in range(min(sp.get_piece_size(), 30)):
        piece = sp.id_to_piece(i)
        score = sp.get_score(i)
        vocab_pieces.append((piece, score))
        print(f"    [{i:>4}] {repr(piece):<20} score={score:.4f}")

    # ── Fertility por idioma (tokens/caractere) ──
    print(f"\n  {'─'*60}")
    print(f"  FERTILITY (tokens por caractere):")
    fertility_tests = {
        "Português": "O gato bebeu leite na cozinha",
        "Inglês":    "The cat drank milk in the kitchen",
        "Japonês":   "猫は台所でミルクを飲みました",
        "Árabe":     "القطة شربت الحليب في المطبخ",
    }
    for lang, text in fertility_tests.items():
        tokens = sp.encode(text, out_type=str)
        fertility = len(tokens) / len(text)
        print(f"    {lang:<12} chars={len(text):>3}  tokens={len(tokens):>3}  "
              f"fertility={fertility:.3f}  [{' | '.join(tokens)}]")

    return results, avg_tokens


def compare_tokenizers(bpe_results, unigram_results, bpe_avg, unigram_avg):
    """
    Comparação lado a lado BPE vs Unigram.
    """
    print(f"\n{'='*70}")
    print(f"  COMPARAÇÃO: BPE vs UNIGRAM")
    print(f"{'='*70}")

    print(f"\n  {'FRASE':<35} {'BPE':>5} {'UNI':>5} {'DIFF':>6} {'VENCEDOR':<10}")
    print(f"  {'─'*35} {'─'*5} {'─'*5} {'─'*6} {'─'*10}")

    bpe_wins = 0
    uni_wins = 0
    ties = 0

    comparison_data = []

    for bpe_r, uni_r in zip(bpe_results, unigram_results):
        b = bpe_r["n_tokens"]
        u = uni_r["n_tokens"]
        diff = b - u
        if diff > 0:
            winner = "UNIGRAM"
            uni_wins += 1
        elif diff < 0:
            winner = "BPE"
            bpe_wins += 1
        else:
            winner = "EMPATE"
            ties += 1

        display = bpe_r["sentence"][:32] + "..." if len(bpe_r["sentence"]) > 35 else bpe_r["sentence"]
        print(f"  {display:<35} {b:>5} {u:>5} {diff:>+6} {winner:<10}")

        comparison_data.append({
            "sentence": bpe_r["sentence"],
            "bpe_tokens": bpe_r["tokens"],
            "bpe_n": b,
            "unigram_tokens": uni_r["tokens"],
            "unigram_n": u,
            "winner": winner,
        })

    print(f"\n  {'─'*70}")
    print(f"  PLACAR FINAL:")
    print(f"    BPE venceu:     {bpe_wins} frases")
    print(f"    Unigram venceu: {uni_wins} frases")
    print(f"    Empates:        {ties} frases")
    print(f"    Média BPE:      {bpe_avg:.1f} tokens/frase")
    print(f"    Média Unigram:  {unigram_avg:.1f} tokens/frase")
    print(f"  {'─'*70}")

    if uni_wins > bpe_wins:
        print(f"  🏆 VENCEDOR GERAL: UNIGRAM (mais compacto)")
    elif bpe_wins > uni_wins:
        print(f"  🏆 VENCEDOR GERAL: BPE (mais compacto)")
    else:
        print(f"  🏆 EMPATE GERAL")

    return comparison_data


def save_results(comparison_data):
    """Salva resultados em JSON."""
    output_path = RESULTS_DIR / "comparison_bpe_vs_unigram.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, ensure_ascii=False, indent=2)
    print(f"\n  ✓ Resultados salvos em: {output_path}")


def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   █████╗ ███████╗████████╗██╗  ██╗███████╗██████╗                ║
║  ██╔══██╗██╔════╝╚══██╔══╝██║  ██║██╔════╝██╔══██╗               ║
║  ███████║█████╗     ██║   ███████║█████╗  ██████╔╝               ║
║  ██╔══██║██╔══╝     ██║   ██╔══██║██╔══╝  ██╔══██╗               ║
║  ██║  ██║███████╗   ██║   ██║  ██║███████╗██║  ██║               ║
║  ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝               ║
║                                                                  ║
║          TOKENIZER LAB — BPE vs UNIGRAM                          ║
║          SentencePiece Comparison Engine                          ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    # Verificar corpus
    if not CORPUS_PATH.exists():
        print(f"[!] Corpus não encontrado: {CORPUS_PATH}")
        sys.exit(1)

    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(f"[✓] Corpus carregado: {len(lines)} linhas, {sum(len(l) for l in lines)} chars")

    # Instalar sentencepiece
    if not install_sentencepiece():
        sys.exit(1)

    # Treinar BPE
    bpe_model = train_tokenizer("bpe", VOCAB_SIZE)

    # Treinar Unigram
    unigram_model = train_tokenizer("unigram", VOCAB_SIZE)

    # Analisar ambos
    bpe_results, bpe_avg = analyze_tokenizer(bpe_model, "bpe")
    unigram_results, unigram_avg = analyze_tokenizer(unigram_model, "unigram")

    # Comparar
    comparison = compare_tokenizers(bpe_results, unigram_results, bpe_avg, unigram_avg)

    # Salvar
    save_results(comparison)

    print(f"\n{'='*70}")
    print(f"  EXECUÇÃO COMPLETA.")
    print(f"  Modelos em: {MODELS_DIR}")
    print(f"  Resultados em: {RESULTS_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
