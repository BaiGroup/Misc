from pyzotero import zotero
import re

# 1. connect
LIB_ID      = ""
LIB_TYPE    = "user"
API_KEY     = ""        # write access needed
zot = zotero.Zotero(LIB_ID, LIB_TYPE, API_KEY)
items = zot.everything(zot.items(itemType="journalArticle"))

# 2. edit
find = "Current Opinion in Solid State and Materials Science"
replace = "Current Opin. Solid State Mater. Sci."
to_update = []
for it in items:
    abbr = it["data"].get("journalAbbreviation", "")
    new_abbr = re.sub(find, replace, abbr)
    if new_abbr != abbr:
        it["data"]["journalAbbreviation"] = new_abbr
        to_update.append(it)
        print(f"Updated {it['data']['title']} to {new_abbr}")

# 3. push
for chunk_start in range(0, len(to_update), 50):
    zot.update_items(to_update[chunk_start:chunk_start+50])
print(f"Updated {len(to_update)} items.")
