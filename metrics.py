# le fichier montre 
"""Métriques Prometheus (exposées sur /metrics). Les étiquettes ont peu de valeurs possibles
(mode, issue, outil, type d'erreur) pour ne pas faire exploser le nombre de séries."""
import time

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

CONTENT_TYPE = CONTENT_TYPE_LATEST

REQUETES = Counter("rag_requetes_total", "Requêtes de chat terminées", ["mode", "issue"])
DUREE = Histogram("rag_requete_duree_secondes", "Durée d'une réponse complète", ["mode"],
                  buckets=(0.5, 1, 2, 3, 5, 8, 12, 20, 30, 60, 120))
EN_COURS = Gauge("rag_requetes_en_cours", "Requêtes en cours de traitement")
ERREURS = Counter("rag_erreurs_total", "Erreurs par type d'exception", ["type"])
SOURCES = Histogram("rag_sources_par_requete", "Sources renvoyées par requête",
                    buckets=(0, 1, 2, 3, 4, 6, 8, 12, 16))
LLM_APPELS = Counter("rag_llm_appels_total", "Appels au LLM de l'agent")
TOKENS = Counter("rag_tokens_total", "Tokens du LLM de l'agent (estimés, réels en entrée, réels en sortie)", ["type"])
OUTIL_APPELS = Counter("rag_outil_appels_total", "Appels d'outils de l'agent", ["outil"])
OUTIL_DUREE = Histogram("rag_outil_duree_secondes", "Durée d'un appel d'outil", ["outil"],
                        buckets=(0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30))
CONTEXTE_TRONQUE = Counter("rag_contexte_resultats_tronques_total", "Anciens résultats d'outils tronqués pour tenir le budget")
CONTEXTE_OUBLIE = Counter("rag_contexte_echanges_oublies_total", "Échanges d'historique oubliés pour tenir le budget")

ISSUES = {"ok": "ok", "erreur": "erreur", "délai dépassé": "delai"}


class Suivi:
    """Transforme les événements de l'agent (call, result, usage, sources) en métriques."""

    def __init__(self):
        self.debuts: dict[str, list[float]] = {}
        self.sources = 0

    def evenement(self, kind, data):
        if kind == "call":
            OUTIL_APPELS.labels(data["tool"]).inc()
            self.debuts.setdefault(data["tool"], []).append(time.perf_counter())
        elif kind == "result":
            file = self.debuts.get(data["tool"])
            if file:
                OUTIL_DUREE.labels(data["tool"]).observe(time.perf_counter() - file.pop(0))
        elif kind == "usage":
            LLM_APPELS.inc()
            TOKENS.labels("estime").inc(data.get("apres") or 0)
            TOKENS.labels("reel_entree").inc(data.get("reel_entree") or 0)
            TOKENS.labels("reel_sortie").inc(data.get("reel_sortie") or 0)
            CONTEXTE_TRONQUE.inc(data.get("outils_tronques") or 0)
            CONTEXTE_OUBLIE.inc(data.get("echanges_perdus") or 0)
        elif kind == "sources":
            self.sources += len(data)


def fin_requete(mode: str, issue: str, duree: float, nb_sources: int) -> None:
    EN_COURS.dec()
    REQUETES.labels(mode, ISSUES.get(issue, issue)).inc()
    DUREE.labels(mode).observe(duree)
    SOURCES.observe(nb_sources)


def exporter() -> bytes:
    return generate_latest()

# Séries créées au démarrage : sans cela, un compteur à étiquettes n'existe qu'après son premier
# incrément, et rate()/increase() ne renvoient rien tant qu'il n'y a pas deux relevés différents.
for _mode in ("direct", "agent"):
    DUREE.labels(_mode)
    for _issue in ("ok", "erreur", "delai"):
        REQUETES.labels(_mode, _issue)
for _outil in ("rechercher_sources", "analyser_image"):
    OUTIL_APPELS.labels(_outil)
    OUTIL_DUREE.labels(_outil)
for _type in ("estime", "reel_entree", "reel_sortie"):
    TOKENS.labels(_type)
