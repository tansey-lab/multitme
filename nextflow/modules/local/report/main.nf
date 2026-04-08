process REPORT {
    tag "$meta.id"
    label 'process_medium'

    container "${params.container_registry}/multitme:${params.container_version}"

    input:
    tuple val(meta), path(predictions_h5ad), path(probs_npy), path(latent_npy), path(scrna_h5ad), path(xenium_h5ad), path(celltype_counts_json)

    output:
    tuple val(meta), path("${meta.id}_cell_type_summary.pdf"),     emit: summary_pdf
    tuple val(meta), path("${meta.id}_confidence_analysis.pdf"),   emit: confidence_pdf
    tuple val(meta), path("${meta.id}_spatial_plots.pdf"),         emit: spatial_pdf
    tuple val(meta), path("${meta.id}_transcript_vs_classification.pdf"), emit: transcript_pdf
    tuple val(meta), path("${meta.id}_xenium_umap.pdf"),            emit: xenium_umap_pdf
    tuple val(meta), path("${meta.id}_scrna_umap.pdf"),             emit: scrna_umap_pdf
    tuple val(meta), path("${meta.id}_report.html"),               emit: report_html
    tuple val(meta), path("${meta.id}_gene_overlap_common_genes.txt"), emit: gene_overlap_genes
    tuple val(meta), path("${meta.id}_choropleth.html"),           emit: choropleth
    path "versions.yml",                                            emit: versions

    script:
    def args = task.ext.args ?: ''
    """
    multitme-report \\
        --input ${predictions_h5ad} \\
        --probs ${probs_npy} \\
        --latent ${latent_npy} \\
        --scrna ${scrna_h5ad} \\
        --xenium ${xenium_h5ad} \\
        --celltype-counts ${celltype_counts_json} \\
        --sample-prefix "${meta.id}_" \\
        --annotation-column ${params.annotation_column ?: 'major_annotation'} \\
        --output-dir . \\
        ${args}

    mv cell_type_summary.pdf ${meta.id}_cell_type_summary.pdf
    mv confidence_analysis.pdf ${meta.id}_confidence_analysis.pdf
    mv spatial_plots.pdf ${meta.id}_spatial_plots.pdf
    mv transcript_vs_classification.pdf ${meta.id}_transcript_vs_classification.pdf
    mv xenium_umap.pdf ${meta.id}_xenium_umap.pdf
    mv scrna_umap.pdf ${meta.id}_scrna_umap.pdf
    mv report.html ${meta.id}_report.html
    mv gene_overlap_common_genes.txt ${meta.id}_gene_overlap_common_genes.txt
    mv choropleth.html ${meta.id}_choropleth.html

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        multitme: \$(python3 -c "import multitme; print(multitme.__version__)")
        matplotlib: \$(python3 -c "import matplotlib; print(matplotlib.__version__)")
    END_VERSIONS
    """
}
