from ogb.lsc import MAG240MDataset
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

dataset = MAG240MDataset()
dataset.num_authors

print(dataset.num_papers) # number of paper nodes
print(dataset.num_authors) # number of author nodes
print(dataset.num_institutions) # number of institution nodes
print(dataset.num_paper_features) # dimensionality of paper features
print(dataset.num_classes) # number of subject area classes

