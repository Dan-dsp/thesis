import os

bird_species = [
    "Anisognathus somptuosus",
    "Atlapetes albinucha",
    "Aulocoyhinchus prasinus",
    "Buteo platipterus",
    "Chlorophanes spiza",
    "Chlorophonia cyanea",
    "Chlorostilbon melanorhynchus",
    "Cinclus leucochephalus",
    "Colibrí delphinae",
    "Cranioleuca erythrops",
    "Eubuco boucerii",
    "Había critata",
    "Heliodoxa rubinoides",
    "Leuconotopicus fumigatus",
    "Melanerpes rubricapillus",
    "Momotus aequatorialis",
    "Myiodynastes chrysocephalus",
    "Myiothlypis fulvicauda",
    "Phaethornis guy",
    "Psittacara wagleri",
    "Pyrrohomyias cinnamomeus",
    "Ramphocelus flamigerus",
    "Rupícola peruviana",
    "Saltator striatipectus",
    "Sayornis nigricans",
    "Sporothraupis cyanicolis",
    "Stilpnia cyanicolis",
    "Stilpnia heinei",
    "Tangara ruficervix",
    "Tangara xanthocephala",
    "Tangara arthurs",
    "Thalurania colombica",
    "Thraupis episcopus",
    "Thraupis palmarum",
    "Troglodytes aedon",
    "Trogon collaris",
    "Trogon personatus",
    "Zonotrichia capensis"
]

# Windows path to your dataset
base_path = r"F:\dataset"

os.makedirs(base_path, exist_ok=True)

for species in bird_species:
    folder_name = species.replace(" ", "_")
    path = os.path.join(base_path, folder_name)
    os.makedirs(path, exist_ok=True)

print("All folders created successfully in F:\\dataset")
