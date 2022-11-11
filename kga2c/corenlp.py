#%%
from dataclasses import dataclass
from typing import NamedTuple, Any, List
import stanza
from stanza.server import CoreNLPClient
import diskcache
import atexit

openie_cache = diskcache.Cache("./.cache/openie")


stanza.install_corenlp()

corenlp_client = CoreNLPClient(
    annotators=["pos", "openie"], timeout=30000, endpoint="http://localhost:8002"
)

corenlp_client.ensure_alive()
corenlp_client.annotate("Init")

#%%


def exit_handler():
    corenlp_client.stop()


atexit.register(exit_handler)


class RDFTriple(NamedTuple):
    subject: str
    relation: str
    object: str
    confidence: float


class Token(NamedTuple):
    word: str
    value: str
    before: str
    after: str
    originalText: str
    lemma: str
    beginChar: int
    endChar: int
    tokenBeginIndex: int
    tokenEndIndex: int
    isNewline: bool


class AnnotatedSentence(NamedTuple):
    rdf_triples: List[RDFTriple]
    tokens: List[Token]


@openie_cache.memoize()
def corenlp_annotate(sentence: str) -> List[AnnotatedSentence]:
    result = corenlp_client.annotate(sentence)  # type: Any
    sentences: List[AnnotatedSentence] = []
    for sentence in result.sentence:
        rdf_triples = list(
            map(
                lambda t: RDFTriple(
                    subject=t.subject,
                    relation=t.relation,
                    object=t.object,
                    confidence=t.confidence,
                ),
                sentence.openieTriple,  # type: ignore
            )
        )
        tokens = list(
            map(
                lambda t: Token(
                    word=t.word,
                    value=t.value,
                    before=t.before,
                    after=t.after,
                    originalText=t.originalText,
                    lemma=t.lemma,
                    beginChar=t.beginChar,
                    endChar=t.endChar,
                    tokenBeginIndex=t.tokenBeginIndex,
                    tokenEndIndex=t.tokenEndIndex,
                    isNewline=t.isNewline,
                ),
                sentence.token,  # type: ignore
            )
        )
        sentences.append(AnnotatedSentence(tokens=tokens, rdf_triples=rdf_triples))

    return sentences
