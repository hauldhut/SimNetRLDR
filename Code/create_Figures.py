import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# === Paths to the image files ===
output_dir = "performance_heatmaps_size_epochs_filter"

# Figure3
figure = "Figure3_AUROCnAUPRC.pdf"
img_paths = [
    os.path.join(output_dir, "OMIM_gat_auroc_heatmap.png"),
    os.path.join(output_dir, "OMIM_gat_auprc_heatmap.png"),
    os.path.join(output_dir, "GeneNet_gat_auroc_heatmap.png"),
    os.path.join(output_dir, "GeneNet_gat_auprc_heatmap.png"),
]

# Figure4
figure = "Figure4_AUROCnAUPRC.pdf"
img_paths = [
    os.path.join(output_dir, "OG_gat_auroc_heatmap.png"),
    os.path.join(output_dir, "OG_gat_auprc_heatmap.png"),
    os.path.join(output_dir, "OHG_gat_auroc_heatmap.png"),
    os.path.join(output_dir, "OHG_gat_auprc_heatmap.png"),
]

# === Read all images ===
images = [mpimg.imread(p) for p in img_paths]

# Assuming all images have the same dimensions
h, w, _ = images[0].shape
aspect_ratio = w / float(h)

# === Labels for subplots ===
labels = ["(a)", "(b)", "(c)", "(d)"]

# === Create figure ===
fig, axes = plt.subplots(2, 2, figsize=(14, 14 / aspect_ratio), gridspec_kw={"hspace": 0, "wspace": 0})
axes = axes.flatten()

# === Plot each image with custom font ===
for ax, img, label in zip(axes, images, labels):
    ax.imshow(img)
    ax.axis("off")
    ax.text(
        0.03, 0.97,
        label,
        transform=ax.transAxes,
        fontsize=20,
        fontweight="bold",
        fontname="Times New Roman",
        va="top",
        ha="left"
    )

# === Adjust layout and save ===
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
output_path = os.path.join(output_dir, figure)
plt.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0)
plt.show()

print(f"Combined figure saved to {output_path}")
