process PREPROCESS {
    tag "$meta.id"
    label 'process_high'

    container "${params.container_registry}/multitme:${params.container_version}"

    input:
    tuple val(meta), path(scrna), path(xenium)

    output:
    tuple val(meta), path("${meta.id}_scrna_filtered.h5ad"),     emit: scrna_adata
    tuple val(meta), path("${meta.id}_xenium_filtered.h5ad"),    emit: xenium_adata
    tuple val(meta), path("${meta.id}_scrna_celltype_counts.json"), emit: celltype_counts
    path "versions.yml",                                         emit: versions

    script:
    def args = task.ext.args ?: ''
    """
    multitme-preprocess \\
        --scrna ${scrna} \\
        --xenium ${xenium} \\
        data.preprocess_method=${params.preprocess_method ?: 'log1p'} \\
        data.annotation_column=${params.annotation_column ?: 'major_annotation'} \\
        output.dir=. \\
        ${args}

    # Rename outputs with sample prefix
    mv scrna_filtered.h5ad ${meta.id}_scrna_filtered.h5ad
    mv xenium_filtered.h5ad ${meta.id}_xenium_filtered.h5ad
    mv scrna_celltype_counts.json ${meta.id}_scrna_celltype_counts.json

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        multitme: \$(python3 -c "import multitme; print(multitme.__version__)")
    END_VERSIONS
    """
}
