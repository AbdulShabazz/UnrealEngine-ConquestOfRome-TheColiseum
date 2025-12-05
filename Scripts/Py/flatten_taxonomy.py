import json

def flatten_taxonomy(data, prefix=""):
    paths = []
    for item in data:
        if isinstance(item, dict):
            for key, value in item.items():
                if isinstance(value, list):
                    for entry in value:
                        if isinstance(entry, dict) and "name" in entry:
                            current_path = f"{prefix}__{key}__{entry['name']}" if prefix else f"{key}__{entry['name']}"
                            paths.append(current_path)
                            # Recurse into subcategories
                            if "subcategories" in entry:
                                for sub in entry["subcategories"]:
                                    sub_path = f"{current_path}__{sub['name']}"
                                    paths.append(sub_path)
    return paths

with open("crc_audio.json","r",encoding="utf-8") as f:
    data = json.load(f)

flat = flatten_taxonomy(data)

# Load JSON (utf-8 is the de-facto default for modern projects)
with open("crc_index.json","w+",encoding="utf-8") as fp:    
    for line in flat:
        fp.write(line + "\n")
print(f"  Written {len(flat)} lines to crc_index.json")
    
"""

**Output:**

soundFxTaxonomy__Ambient Environment
soundFxTaxonomy__Ambient Environment__Coliseum Ambient
soundFxTaxonomy__Ambient Environment__City / Marketplace Ambient
soundFxTaxonomy__Footsteps & Movement
soundFxTaxonomy__Footsteps & Movement__Stone / Marble Floor
soundFxTaxonomy__Footsteps & Movement__Sand / Dust
soundFxTaxonomy__Footsteps & Movement__Wooden Deck / Bridge
soundFxTaxonomy__Footsteps & Movement__Grass / Meadow
soundFxTaxonomy__Footsteps & Movement__Water / Shallow Stream
"""