"""Source unique des outils de l'agent.

Ces trois fonctions sont exposées telles quelles par le serveur MCP (mcp_server.py) et
par l'agent en mode local (agent.py). Leurs docstrings sont les descriptions que le LLM lit
pour choisir un outil : on ne les écrit qu'une fois.

Ce module reste léger : rag_core est importé à l'appel, pas à l'import. Ainsi, en mode MCP,
le processus de l'agent connaît les noms des outils sans charger CLIP ni la base vectorielle.
"""


def rechercher_sources(question: str, k: int = 6, type_resultat: str = "") -> str:
    """Interroge la base vectorielle et renvoie les passages de texte et les images les plus proches
    (id, pdf, page, contenu, similarité), SANS rédiger de réponse. La base est indexée en ANGLAIS :
    formule la requête en anglais. type_resultat : "texte", "image", ou chaîne vide pour les deux."""
    import rag_core
    return rag_core.outil_rechercher(question, k, type_resultat)


def analyser_image(id_image: str, question: str) -> str:
    """Lit une image trouvée par rechercher_sources (schéma, graphique, tableau) et renvoie une
    observation en texte. id_image = champ 'id' d'une source de type image."""
    import rag_core
    return rag_core.outil_analyser_image(id_image, question)


def repondre_question(question: str, k: int = 6, type_resultat: str = "") -> str:
    """Pipeline RAG complet : recherche puis réponse rédigée par un modèle, avec ses sources.
    Réservé aux clients MCP externes : l'agent de l'application rédige lui-même sa réponse."""
    import rag_core
    return rag_core.outil_repondre(question, k, type_resultat)


OUTILS = (rechercher_sources, analyser_image, repondre_question)
OUTILS_AGENT = frozenset({"rechercher_sources", "analyser_image"})
