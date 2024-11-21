import re
def extract_genus_species(sci_name):
    pattern= r"^(\w+)\s+(\w+)$"
    match=re.match(pattern, sci_name)
    if match:
        genus,species = match.groups()
        print(f"Genus:{genus}")
        print(f"Species:{species}")
    else:
        print("Invalid scientific name format.")
sci_name1="Homo sapiens"
sci_name2="Drosophila melanogaster"
sci_name3="Aspergillus niger"
extract_genus_species(sci_name1)
extract_genus_species(sci_name2)
extract_genus_species(sci_name3)