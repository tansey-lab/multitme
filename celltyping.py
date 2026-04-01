import numpy as np
import torch
from torch.utils.data import DataLoader
import scanpy as sc
from multitme_simple_sc_xenium import MultiModalCycleVAE, CyclingDataset, CycleVAETrainer, preprocess
from pseudo_label_markers import pseudo_label_from_markers

scRNA_train = sc.read_h5ad('scRNA_sample.h5ad')
transcriptome_train = sc.read_h5ad('xenium_sample.h5ad')

scRNA_train = scRNA_train[scRNA_train.X.sum(axis=1) > 0]
scRNA_data = preprocess(np.array(scRNA_train.X.todense()), 'clr')
transcriptome_train = transcriptome_train[transcriptome_train.X.sum(axis=1) > 0]
transcriptome_data = preprocess(np.array(transcriptome_train.X.todense()), 'clr')

# common genes for common-gene alignment (only used in scrna+xenium)
common_genes = np.intersect1d(scRNA_train.var_names, transcriptome_train.var_names)
indices_scrna = scRNA_train.var.index.get_indexer(common_genes)
indices_xenium = transcriptome_train.var.index.get_indexer(common_genes)

DEVICE = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)
SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)

# cell type annotations
unique_types = sorted(set(scRNA_train.obs['major_annotation']))
type_to_idx = {t: i for i, t in enumerate(unique_types)}
print(f"Cell types ({len(unique_types)}): {unique_types}")

# scrna with cell type annotation
scRNA_label_tensor = torch.tensor(
    [type_to_idx[t] for t in scRNA_train.obs['major_annotation']], dtype=torch.long
)
# xenium pseudo labels with marker genes
marker_dict = {
    'B': ['CD19', 'MS4A1', 'CD79A', 'CD22', 'POU2AF1'],
    'CD4T': ['CD3E', 'CD2', 'CD247', 'CTLA4', 'JAK3'],
    'CD8T': ['CD8A', 'CD3E', 'GZMA', 'GZMH', 'ITGAL'],
    'Endothelial': ['HSPG2', 'COL4A1', 'PLVAP', 'PECAM1', 'COL4A2'],
    'Epithelial': ['EPCAM', 'ST14', 'IGF2', 'COL2A1', 'CDH1'],
    'Fibroblast': ['COL5A1', 'COL5A2', 'POSTN', 'MMP14', 'CTSK'],
    'Macrophage': ['CD14', 'STAB1', 'FCGR2A', 'ITGB2', 'CD4'],
    'Mast': ['HDC', 'GATA2', 'KIT', 'IL1RL1', 'SLC18A2'],
    # 'Monocyte': ['PLAUR', 'CD14', 'IL1B', 'CD300E', 'ITGAX'],
    'Neutrophil': ['CSF3R', 'ITGAX', 'TREM1', 'HCAR2', 'BCL2A1'],
    'Plasma': ['MZB1', 'DERL3', 'PIM2', 'XBP1', 'TENT5C'],
    'SMC': ['COL4A1', 'COL18A1', 'MCAM', 'COL4A2', 'PDGFRB'],
    'Tprolif': ['CD3E', 'TUBB'],
    # 'cDC2': ['ITGB2', 'CD4', 'SAMHD1'],
    'mregDC': ['FSCN1', 'CCL22', 'LAMP3', 'CD83', 'LY75'],
    # 'pDC': ['CIITA', 'GZMB', 'IRF8', 'IRF4', 'CD4']
}
xenium_gene_names = transcriptome_train.var_names
xenium_label_tensor = pseudo_label_from_markers(
    data=transcriptome_data,
    gene_names=xenium_gene_names,
    marker_dict=marker_dict,
    type_to_idx=type_to_idx,
    top_k=50,
    normalize=False
)

scRNA_tensor = torch.tensor(scRNA_data, dtype=torch.float32)
xenium_tensor = torch.tensor(transcriptome_data, dtype=torch.float32)
print(f"scRNA:  {scRNA_tensor.shape}  (labeled: {(scRNA_label_tensor >= 0).sum().item()})")
print(f"Xenium: {xenium_tensor.shape}  (labeled: {(xenium_label_tensor >= 0).sum().item()})")

model = MultiModalCycleVAE(
    modality_dims={'scrna': scRNA_tensor.shape[1], 'xenium': xenium_tensor.shape[1]},
    n_latent=20,
    hidden_dims=[512, 256],
    common_masks={'scrna': indices_scrna, 'xenium': indices_xenium},
    cycle_pairs=[('scrna', 'xenium'), ('xenium', 'scrna')],
    n_cell_types=len(unique_types),
    aux_loss_multiplier=1000.0,
    type_alignment_weight=100.0,
    alignment_method='swd',
    cycle_cls_weight=1000.0,
    labeled_modality='scrna'
)
model = model.to(DEVICE)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"aux_loss_multiplier: {model.aux_loss_multiplier}, type_alignment_weight: {model.type_alignment_weight}, cycle_pairs: {model.cycle_pairs}")

dataset = CyclingDataset(
    modality_dict={'scrna': scRNA_tensor, 'xenium': xenium_tensor},
    label_dict={'scrna': scRNA_label_tensor, 'xenium': xenium_label_tensor},
    target_batch_size=4096,
)
loader = DataLoader(dataset, batch_size=None, shuffle=False)

trainer = CycleVAETrainer(
    model, learning_rate=1e-3,
    cycle_weight=1.0,
    beta=1.0,
    beta_warmup_epochs=10,
)
celltype_gt = transcriptome_train.obs['annotation'].map(type_to_idx)
history = trainer.fit(loader, n_epochs=50, print_every=5)

model.eval()
 
with torch.no_grad():
    # latent embeddings
    z_scrna = model.get_latent(scRNA_tensor.to(DEVICE), 'scrna').cpu().numpy()
    z_xenium = model.get_latent(xenium_tensor.to(DEVICE), 'xenium').cpu().numpy()
 
    # predict cell types for Xenium
    xenium_probs = model.predict_cell_types(xenium_tensor.to(DEVICE), 'xenium').cpu().numpy()
    xenium_pred_idx = xenium_probs.argmax(axis=1)
    xenium_pred_types = np.array([unique_types[i] for i in xenium_pred_idx])
 
    # optional panel completion
    xenium_as_scrna = model.translate(xenium_tensor.to(DEVICE), 'xenium', 'scrna').cpu().numpy()
    
print((xenium_pred_idx == celltype_gt).sum() / transcriptome_train.shape[0])
np.save('z_scrna_20', z_scrna)
np.save('z_xenium_20', z_xenium)
np.save('xenium_pred_idx_20', xenium_pred_idx)
# np.save('xenium_as_scrna', xenium_as_scrna)
# import umap
# from sklearn.preprocessing import StandardScaler
# z_combined = np.vstack([z_scrna, z_xenium])
# scaler = StandardScaler()
# z_combined_std = scaler.fit_transform(z_combined)
# reducer = umap.UMAP()
# embedding_combined = reducer.fit_transform(z_combined_std)

# n_X = len(z_scrna)
# embedding_x = embedding_combined[:n_X]
# embedding_y = embedding_combined[n_X:]
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
# import matplotlib.patches as mpatches
# Glasbey = [
#    "#0000FF",  # blue
#    "#FF0000",  # red
#    "#FF00B6",  # magenta
#    "#000033",  # dark navy
#    "#00FF00",  # green
#    "#005300",  # dark green
#    "#FFD300",  # yellow
#    "#009FFF",  # sky blue
#    "#9A4D42",  # brown
#    "#00FFBE",  # turquoise
#    "#783FC1",  # purple
#    "#1F9698",  # teal
#    "#FF7A5C",  # coral
#    "#4A3B53",  # deep purple-gray
#    "#FE8F42",  # orange
#    "#A6BDD7",  # light steel blue
#    "#B0FF9D",  # light green
#    "#C20088",  # deep magenta
#    "#003380",  # deep blue
#    "#FFA405",  # orange-yellow
# ]
# fig, ax = plt.subplots(1, 3, figsize=(15, 8), sharex=True, sharey=True)
# for i in range(16):
#    xenium_filter = transcriptome_train.obs['annotation'] == labels[i]
#    scrna_filter = scRNA_train.obs['major_annotation'] == labels[i]
#    pred_filter = xenium_pred_idx == i
#    ax[1].scatter(embedding_y[xenium_filter, 1], embedding_y[xenium_filter, 0], s=0.1, edgecolor='none', c=Glasbey[i])
#    ax[0].scatter(embedding_x[scrna_filter, 1], embedding_x[scrna_filter, 0], s=1, edgecolor='none', c=Glasbey[i])
#    ax[2].scatter(embedding_y[pred_filter, 1], embedding_y[pred_filter, 0], s=0.1, edgecolor='none', c=Glasbey[i])

# ax[0].set_aspect('equal', adjustable='box')
# ax[1].set_aspect('equal', adjustable='box')
# ax[2].set_aspect('equal', adjustable='box')
# ax[0].set_axis_off()
# ax[1].set_axis_off()
# ax[2].set_axis_off()
# plt.tight_layout()
# plt.show()
