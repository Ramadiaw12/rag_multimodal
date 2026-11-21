# Fichier pour appoter l'ingénierie de contexte de l'agent, avec filtrage du retrieval, compression des passages, fenêtre de conversation et ajustement au budget de tokens. 
# Ces fonctions sont utilisées par l'agent et le serveur MCP pour préparer les messages envoyés au LLM et gérer le contexte de la conversation.
"""Ingénierie de contexte de l'agent.

1. Retrieval filtering    -> filtrer_sources()
2. Context compression    -> compresser_passage() + troncature des vieux résultats d'outils
3. Conversation window    -> fenetre()
4. Token counting         -> count_tokens(), count_messages(), ajuster_au_budget()

Tous les seuils se règlent par variables d'environnement (.env), sans toucher au code.
"""
import json
import os
import re

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", 6000))   # budget par appel LLM
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", 3))                # échanges précédents conservés
MARGE_RELATIVE = float(os.getenv("MARGE_RELATIVE", 0.08))         # écart max avec le meilleur score
SIM_MIN_TEXTE = float(os.getenv("SIM_MIN_TEXTE", 0.0))            # seuils absolus (à calibrer)
SIM_MIN_IMAGE = float(os.getenv("SIM_MIN_IMAGE", 0.0))
MAX_CHARS_PASSAGE = int(os.getenv("MAX_CHARS_PASSAGE", 450))      # taille max d'un passage envoyé

# ---------------------------------------------------------------- 4. Comptage de tokens
try:  # estimation avec le tokenizer o200k (famille gpt-oss) ; repli sur ~4 caractères par token
    import tiktoken
    _enc = tiktoken.get_encoding("o200k_base")

    def count_tokens(texte: str) -> int:
        return len(_enc.encode(texte or ""))
except Exception:
    def count_tokens(texte: str) -> int:
        return len(texte or "") // 4 + 1

# On a definit les fonctions pour filtrer les sources de retrieval, compresser les passages de texte, gérer la fenêtre de conversation et ajuster les messages au budget de tokens. Ces fonctions sont utilisées par l'agent et le serveur MCP pour préparer les messages envoyés au LLM et gérer le contexte de la conversation.
def _contenu(m) -> str:
    c = m.content
    if isinstance(c, str):
        return c
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)

# ici count_messages calcule le nombre total de tokens dans une liste de messages, en comptant les tokens de balisage et les tokens des appels d'outils. Il est utilisé pour ajuster les messages au budget de tokens lors de l'envoi au LLM.
def count_messages(messages) -> int:
    total = 0
    for m in messages:
        total += 4 + count_tokens(_contenu(m))   # ~4 tokens de balisage par message
        for tc in getattr(m, "tool_calls", None) or []:
            total += count_tokens(tc.get("name", "")) + count_tokens(json.dumps(tc.get("args", {})))
    return total


# ---------------------------------------------------------------- 1. Filtrage du retrieval
# filtrer_sources garde uniquement les sources utiles en appliquant un seuil absolu par type, un écart maximal avec le meilleur score du même type, puis en dédoublonnant les résultats. Les sources sont renvoyées sous forme de liste de dictionnaires.
def filtrer_sources(sources: list[dict]) -> list[dict]:
    """Ne garde que les sources utiles : seuil absolu par type, écart maximal avec le meilleur
    score du même type (les échelles texte et image diffèrent), puis dédoublonnage."""
    meilleur: dict = {}
    for s in sources:
        meilleur[s["type"]] = max(meilleur.get(s["type"], -1.0), s["similarite"])
    gardees, vus = [], set()
    for s in sources:
        t = s["type"]
        seuil = max(SIM_MIN_TEXTE if t == "texte" else SIM_MIN_IMAGE, meilleur[t] - MARGE_RELATIVE)
        if s["similarite"] < seuil:
            continue
        cle = (s["pdf"], s["page"], (s["contenu"] or "")[:80]) if t == "texte" else s["id"]
        if cle in vus:
            continue
        vus.add(cle)
        gardees.append(s)
    return gardees


# ---------------------------------------------------------------- 2. Compression
# compresse un passage de texte en ne gardant que les phrases qui partagent le plus de mots avec la requête, dans leur ordre d'origine, jusqu'à un nombre maximal de caractères. Aucun appel LLM supplémentaire n'est effectué.
_STOP = set("the and for are was with that this from have not but you your les des une est dans pour que qui sur par "
            "pas aux elle ils son ses mais comme plus avec cette sont ont fait être".split())

# mots extrait les mots significatifs d'un texte, en ignorant les mots vides et les mots de moins de 3 caractères. Il est utilisé pour comparer les phrases d'un passage avec la requête afin de déterminer quelles phrases conserver lors de la compression
def _mots(texte: str) -> set:
    return {w for w in re.findall(r"\w+", texte.lower()) if len(w) > 2 and w not in _STOP}

# fonction qui compresse un passage de texte en ne gardant que les phrases qui partagent le plus de mots avec la requête, dans leur ordre d'origine, jusqu'à un nombre maximal de caractères. Aucun appel LLM supplémentaire n'est effectué. Elle est utilisée pour réduire la taille des passages envoyés au LLM lors de la génération de réponses.
def compresser_passage(texte: str, requete: str, max_chars: int = MAX_CHARS_PASSAGE) -> str:
    """Compression extractive : garde les phrases qui partagent le plus de mots avec la requête,
    dans leur ordre d'origine, jusqu'à max_chars. Aucun appel LLM supplémentaire."""
    texte = " ".join((texte or "").split())
    if len(texte) <= max_chars:
        return texte
    phrases = re.split(r"(?<=[.!?])\s+", texte)
    q = _mots(requete)
    classement = sorted(range(len(phrases)), key=lambda i: -len(q & _mots(phrases[i])))
    choisies, total = [], 0
    for i in classement:
        if total + len(phrases[i]) > max_chars and choisies:
            continue
        choisies.append(i)
        total += len(phrases[i]) + 1
        if total >= max_chars:
            break
    return " ".join(phrases[i] for i in sorted(choisies))[:max_chars]

# tronque un texte à n caractères et ajoute " …[tronqué]" si le texte est plus long. Cette fonction est utilisée pour réduire la taille des passages ou des résultats d'outils avant de les envoyer au LLM, afin de respecter le budget de tokens.
def _tronquer(texte: str, n: int) -> str:
    return texte if len(texte) <= n else texte[:n] + " …[tronqué]"


# ---------------------------------------------------------------- 3. Fenêtre de conversation
# context window : fenetre garde les derniers échanges de l'historique avant la question courante, ainsi que tout le tour courant. Elle est utilisée pour limiter le contexte envoyé au LLM tout en conservant les informations pertinentes de la conversation.
def fenetre(messages: list, tours: int = HISTORY_TURNS) -> list:
    """Garde les `tours` derniers échanges avant la question courante, puis tout le tour courant."""
    humains = [k for k, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if not humains:
        return list(messages)
    i = humains[-1]
    histo = list(messages[max(0, i - 2 * tours):i])
    while histo and not isinstance(histo[0], HumanMessage):   # jamais de réponse orpheline en tête
        histo.pop(0)
    return histo + list(messages[i:])


# ------------------------------------------------------- Ajustement au budget
# ajuster_au_budget réduit les messages envoyés au LLM pour respecter le budget de tokens. Il tronque les résultats d'outils anciens, supprime les échanges d'historique les plus anciens, et en dernier recours tronque le résultat d'outil le plus récent. Il renvoie les messages ajustés et des statistiques sur l'ajustement.
def ajuster_au_budget(messages: list, systeme: str, surcout_outils: int = 0, budget: int = MAX_CONTEXT_TOKENS):
    """Prépare les messages envoyés au LLM. L'état de l'agent n'est jamais modifié : seule la copie
    envoyée est réduite. Ordre de sacrifice, du moins coûteux au plus coûteux :
      1) résultats d'outils anciens tronqués (le plus récent reste intact),
      2) échanges d'historique supprimés, du plus ancien au plus récent,
      3) en dernier recours, résultat d'outil le plus récent tronqué."""
    msgs = fenetre(messages)
    base = count_tokens(systeme) + surcout_outils
    total = lambda: base + count_messages(msgs)
    avant, tronques, perdus = total(), 0, 0

    outils = [k for k, m in enumerate(msgs) if isinstance(m, ToolMessage)]
    for k in outils[:-1]:
        if total() <= budget:
            break
        msgs[k] = msgs[k].model_copy(update={"content": _tronquer(_contenu(msgs[k]), 300)})
        tronques += 1

    while total() > budget:
        dernier_humain = max(k for k, m in enumerate(msgs) if isinstance(m, HumanMessage))
        if dernier_humain < 2:
            break
        msgs = msgs[2:]
        perdus += 1

    if total() > budget and outils:
        k = max(k for k, m in enumerate(msgs) if isinstance(m, ToolMessage))
        msgs[k] = msgs[k].model_copy(update={"content": _tronquer(_contenu(msgs[k]), 1200)})
        tronques += 1

    return msgs, {"avant": avant, "apres": total(), "budget": budget,
                  "outils_tronques": tronques, "echanges_perdus": perdus}