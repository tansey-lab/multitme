process PREPROCESS {
    tag "$meta.id"
    label 'process_medium'

    container "ghcr.io/${params.container_registry}/multitme:${params.container_version}"

    input:
    tuple val(meta), path(scrna_h5ad), path(xenium_input)

    output:
    tuple val(meta), path("${meta.id}_scrna_preprocessed.npy"),  emit: scrna_data
    tuple val(meta), path("${meta.id}_xenium_preprocessed.npy"), emit: xenium_data
    tuple val(meta), path("${meta.id}_scrna_filtered.h5ad"),     emit: scrna_adata
    tuple val(meta), path("${meta.id}_xenium_filtered.h5ad"),    emit: xenium_adata
    path "versions.yml",                                         emit: versions

    script:
    def args = task.ext.args ?: ''
    """
    multitme-preprocess \\
        data.scrna_path=${scrna_h5ad} \\
        data.xenium_path=${xenium_input} \\
        data.preprocess_method=${params.preprocess_method ?: 'clr'} \\
        output.dir=. \\
        ${args}

    # Rename outputs with sample prefix
    mv scrna_preprocessed.npy ${meta.id}_scrna_preprocessed.npy
    mv xenium_preprocessed.npy ${meta.id}_xenium_preprocessed.npy
    mv scrna_filtered.h5ad ${meta.id}_scrna_filtered.h5ad
    mv xenium_filtered.h5ad ${meta.id}_xenium_filtered.h5ad

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        multitme: \$(python3 -c "import multitme; print(multitme.__version__)")
    END_VERSIONS
    """
}
