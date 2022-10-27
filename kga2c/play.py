#%%
from stanza.server import CoreNLPClient
import stanza

stanza.install_corenlp()

client = CoreNLPClient(
    annotators=["pos", "openie"],
    be_quiet=True,
    timeout=30000,
    endpoint="http://localhost:8002",
)

# %%

result = client.annotate("Software is kinda cool")
# %%
